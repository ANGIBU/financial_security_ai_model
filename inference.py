# inference.py
"""
실행 파일
"""

import os
import sys
import time
import re
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import warnings
import gc
from typing import List, Dict, Tuple, Optional
import threading
import psutil
import platform
warnings.filterwarnings("ignore")

# 플랫폼별 시그널 처리
IS_WINDOWS = platform.system() == "Windows"
if not IS_WINDOWS:
    import signal

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Transformers 관련 import
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import logging
logging.set_verbosity_error()

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("작업 시간 초과")

class CrossPlatformTimeout:
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.is_windows = IS_WINDOWS
        self.timer = None
        self.timed_out = False
    
    def __enter__(self):
        if self.is_windows:
            self.timer = threading.Timer(self.timeout_seconds, self._timeout)
            self.timer.start()
        else:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.timeout_seconds))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_windows:
            if self.timer:
                self.timer.cancel()
            if self.timed_out:
                raise TimeoutException("작업 시간 초과")
        else:
            signal.alarm(0)
    
    def _timeout(self):
        self.timed_out = True

class SOLARInference:
    """추론 엔진"""
    
    def __init__(self, model_name: str = "upstage/SOLAR-10.7B-Instruct-v1.0"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        # 성능 통계
        self.cache = {}
        self.stats = {
            "total_processed": 0,
            "cache_hits": 0,
            "successful": 0,
            "failed": 0
        }
        
        print(f"모델 로딩: {model_name}")
        self._load_model()
        print("모델 로딩 완료")
    
    def _load_model(self):
        try:
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Flash Attention 확인
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
                print("Flash Attention 2 사용")
            except ImportError:
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    attn_implementation = "sdpa"
                    print("Scaled Dot Product Attention 사용")
                else:
                    attn_implementation = "eager"
                    print("Standard Attention 사용")
            
            # GPU 메모리에 따른 설정 조정
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if gpu_memory < 12:  # 8GB GPU 최적화
                print(f"GPU 메모리 {gpu_memory:.1f}GB - 메모리 절약 모드")
                device_map = {"": 0}  # 단일 GPU에 모든 레이어
                max_memory = {0: f"{int(gpu_memory * 0.85)}GB"}
                torch_dtype = torch.float16
            else:
                device_map = "auto"
                max_memory = None
                torch_dtype = torch.float16
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                max_memory=max_memory,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                low_cpu_mem_usage=True
            )
            
            # Generation Config 설정
            self.generation_config = GenerationConfig(
                max_new_tokens=128,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.05,
                no_repeat_ngram_size=2
            )
            
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            raise
    
    def analyze_question(self, question: str) -> Dict:
        """문제 분석"""
        analysis = {
            "is_multiple_choice": self._detect_multiple_choice(question),
            "has_negative": "않" in question or "없" in question or "틀린" in question,
            "keywords": self._extract_keywords(question),
            "complexity": self._estimate_complexity(question)
        }
        return analysis
    
    def _detect_multiple_choice(self, question: str) -> bool:
        """객관식 감지"""
        patterns = [
            r'[①②③④⑤]',
            r'\b[1-5]\s*[.)]',
            r'^\s*[1-5]\s+[가-힣]',
            r'\n\s*[1-5]\s',
            r'[1-5]번',
            r'선택지',
            r'다음.*중.*(?:맞|옳|적절|해당)',
        ]
        
        for pattern in patterns:
            if re.search(pattern, question, re.MULTILINE):
                return True
        
        numbers = re.findall(r'\b[1-5]\b', question)
        if len(set(numbers)) >= 3:
            return True
            
        return False
    
    def _extract_keywords(self, question: str) -> List[str]:
        financial_keywords = [
            "금융", "보안", "개인정보", "암호화", "인증", "전자금융",
            "피싱", "스미싱", "파밍", "보이스피싱", "사기", "해킹",
            "전자서명", "공인인증서", "OTP", "생체인증", "블록체인"
        ]
        
        found_keywords = []
        for keyword in financial_keywords:
            if keyword in question:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _estimate_complexity(self, question: str) -> str:
        if len(question) < 100:
            return "easy"
        elif len(question) < 200:
            return "medium"
        else:
            return "hard"
    
    def create_prompt(self, question: str, analysis: Dict) -> str:
        if analysis["is_multiple_choice"]:
            return self._create_mc_prompt(question, analysis)
        else:
            return self._create_open_prompt(question, analysis)
    
    def _create_mc_prompt(self, question: str, analysis: Dict) -> str:
        """객관식 프롬프트"""
        if analysis["has_negative"]:
            system_msg = "다음 객관식 문제에서 틀린 것 또는 해당하지 않는 것을 찾으세요."
        else:
            system_msg = "다음 객관식 문제의 정답을 선택하세요."
        
        prompt = f"""{system_msg}

{question}

정답 번호:"""
        
        return prompt
    
    def _create_open_prompt(self, question: str, analysis: Dict) -> str:
        """주관식 프롬프트"""
        prompt = f"""다음 질문에 대해 간결하고 정확하게 답변하세요.

{question}

답변:"""
        
        return prompt
    
    def generate_response(self, prompt: str, timeout: int = 10) -> str:
        """응답 생성"""
        
        try:
            # 캐시 확인
            cache_key = hash(prompt[:100])
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]
            
            # 타임아웃 적용
            with CrossPlatformTimeout(timeout):
                # 대화 템플릿 적용
                conversation = [{'role': 'user', 'content': prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # 토크나이징
                inputs = self.tokenizer(
                    formatted_prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=1024
                ).to(self.model.device)
                
                # 생성
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=self.generation_config,
                        use_cache=True
                    )
                
                # 디코딩
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                # 캐시 저장
                self.cache[cache_key] = response
                self.stats["successful"] += 1
                
                return response
                
        except TimeoutException:
            self.stats["failed"] += 1
            return self._get_fallback_answer(prompt)
        except Exception as e:
            print(f"생성 오류: {e}")
            self.stats["failed"] += 1
            return self._get_fallback_answer(prompt)
    
    def _get_fallback_answer(self, prompt: str) -> str:
        if any(pattern in prompt for pattern in ["①", "1)", "1.", "선택"]):
            return "3"  # 객관식 기본값
        else:
            if "개인정보" in prompt:
                return "개인정보보호법에 따른 안전성 확보조치가 필요합니다."
            elif "전자금융" in prompt:
                return "전자금융거래법에 따른 보안대책을 수립해야 합니다."
            else:
                return "관련 법규에 따른 적절한 조치가 필요합니다."
    
    def extract_answer(self, response: str, is_multiple_choice: bool) -> str:
        if is_multiple_choice:
            # 숫자 추출
            numbers = re.findall(r'\b([1-5])\b', response)
            if numbers:
                return numbers[-1]
            
            # 원 번호 추출
            circle_match = re.search(r'[①②③④⑤]', response)
            if circle_match:
                circle_to_num = {'①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5'}
                return circle_to_num[circle_match.group()]
            
            return "3"  # 기본값
        else:
            # 주관식 정리
            cleaned = response.replace("답변:", "").strip()
            if len(cleaned) < 10:
                return self._get_fallback_answer("")
            return cleaned[:300]
    
    def cleanup(self):
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            
            print(f"\n성능 통계:")
            print(f"- 총 처리: {self.stats['total_processed']}개")
            print(f"- 성공: {self.stats['successful']}개")
            print(f"- 실패: {self.stats['failed']}개") 
            print(f"- 캐시 히트: {self.stats['cache_hits']}개")
            
        except Exception as e:
            print(f"정리 중 오류: {e}")

class InferenceEngine:
    """추론 엔진"""
    
    def __init__(self):
        self.start_time = time.time()
        self.model_handler = SOLARInference()
        
        print("초기화 완료")
    
    def execute_inference(self, test_file: str, submission_file: str, 
                         output_file: str = "./final_submission.csv") -> Dict:
        """메인 추론 실행"""
        
        # 데이터 로드
        test_df = pd.read_csv(test_file)
        sample_submission = pd.read_csv(submission_file)
        
        questions = test_df['Question'].tolist()
        question_ids = test_df['ID'].tolist()
        
        print(f"데이터 로드 완료: {len(questions)}개 문항")
        
        # 문제 분석
        print("문제 분석 중...")
        analyzed_questions = []
        for i, question in enumerate(tqdm(questions, desc="분석")):
            analysis = self.model_handler.analyze_question(question)
            analyzed_questions.append({
                "index": i,
                "question": question,
                "analysis": analysis
            })
        
        # 통계 출력
        mc_count = sum(1 for q in analyzed_questions if q["analysis"]["is_multiple_choice"])
        open_count = len(analyzed_questions) - mc_count
        
        print(f"\n=== 문제 분석 완료 ===")
        print(f"총 문항: {len(questions)}")
        print(f"객관식: {mc_count}개")
        print(f"주관식: {open_count}개")
        
        # 객관식 우선 처리
        analyzed_questions.sort(key=lambda x: (
            not x["analysis"]["is_multiple_choice"],
            x["analysis"]["complexity"] == "hard",
            len(x["question"])
        ))
        
        # 추론 실행
        predictions = [""] * len(questions)
        
        print("\n추론 실행 중...")
        for q_info in tqdm(analyzed_questions, desc="추론"):
            idx = q_info["index"]
            question = q_info["question"]
            analysis = q_info["analysis"]
            
            # 타임아웃 설정
            if analysis["is_multiple_choice"]:
                timeout = 8
            else:
                timeout = 15
            
            # 프롬프트 생성
            prompt = self.model_handler.create_prompt(question, analysis)
            
            # 응답 생성
            response = self.model_handler.generate_response(prompt, timeout)
            
            # 답변 추출
            answer = self.model_handler.extract_answer(response, analysis["is_multiple_choice"])
            predictions[idx] = answer
            
            self.model_handler.stats["total_processed"] += 1
            
            # 메모리 정리
            if idx % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # 결과 저장
        sample_submission['Answer'] = predictions
        sample_submission.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 통계 계산
        total_time = time.time() - self.start_time
        mc_answers = [p for p in predictions if p.strip().isdigit()]
        
        # 분포 계산
        distribution = {}
        for answer in mc_answers:
            distribution[answer] = distribution.get(answer, 0) + 1
        
        results = {
            "output_file": output_file,
            "total_questions": len(predictions),
            "total_time_minutes": total_time / 60,
            "mc_count": mc_count,
            "open_count": open_count,
            "answer_distribution": distribution,
            "success": True
        }
        
        # 결과 출력
        print("\n=== 최종 결과 ===")
        print(f"총 처리: {len(predictions)}개")
        print(f"소요 시간: {total_time/60:.1f}분")
        print(f"객관식 {mc_count}개, 주관식 {open_count}개")
        
        if mc_answers:
            print("\n객관식 답변 분포:")
            for choice in sorted(distribution.keys()):
                count = distribution[choice]
                pct = (count / len(mc_answers)) * 100
                print(f"  {choice}번: {count}개 ({pct:.1f}%)")
        
        print(f"\n최종 제출 파일: {output_file}")
        
        return results
    
    def cleanup(self):
        self.model_handler.cleanup()

def main():
    
    print(f"실행 환경: {platform.system()}")
    
    # GPU 확인
    if not torch.cuda.is_available():
        print("경고: CUDA 사용 불가, CPU로 실행")
    else:
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_info.name} ({gpu_info.total_memory / (1024**3):.1f}GB)")
    
    # 파일 확인
    test_file = "./test.csv"
    submission_file = "./sample_submission.csv"
    
    if not os.path.exists(test_file) or not os.path.exists(submission_file):
        print("오류: 데이터 파일 없음")
        print(f"확인 필요: {test_file}, {submission_file}")
        sys.exit(1)
    
    # 추론 실행
    engine = None
    try:
        engine = InferenceEngine()
        results = engine.execute_inference(test_file, submission_file)
        
        if results["success"]:
            print("\n✅ 추론 성공적으로 완료!")
        
    except KeyboardInterrupt:
        print("\n추론 중단")
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if engine:
            engine.cleanup()

if __name__ == "__main__":
    main()