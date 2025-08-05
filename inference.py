# inference.py
"""
실행 파일 - SOLAR 모델 최적화 버전
"""

import os
import sys
import time
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
import re
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
    """Unix 시그널 핸들러"""
    raise TimeoutException("작업 시간 초과")

class CrossPlatformTimeout:
    """크로스 플랫폼 타임아웃 관리자"""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.is_windows = IS_WINDOWS
        self.timer = None
        self.timed_out = False
    
    def __enter__(self):
        if self.is_windows:
            # Windows: threading.Timer 사용
            self.timer = threading.Timer(self.timeout_seconds, self._timeout)
            self.timer.start()
        else:
            # Linux: signal.alarm 사용
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
            signal.alarm(0)  # 타임아웃 해제
    
    def _timeout(self):
        """Windows용 타임아웃 콜백"""
        self.timed_out = True

class OptimizedSOLARInference:
    """SOLAR 모델 최적화 추론 엔진"""
    
    def __init__(self, model_name: str = "upstage/SOLAR-10.7B-Instruct-v1.0"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        # 성능 통계
        self.cache = {}  # 간단한 캐시
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
        """모델 로드"""
        try:
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Flash Attention 2 사용 가능 여부 확인
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
                print("Flash Attention 2 사용")
            except ImportError:
                # PyTorch 2.0+ SDPA 사용 (더 효율적)
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    attn_implementation = "sdpa"
                    print("Scaled Dot Product Attention 사용")
                else:
                    attn_implementation = "eager"
                    print("Standard Attention 사용")
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                attn_implementation=attn_implementation
            )
            
            # Generation Config 설정
            self.generation_config = GenerationConfig(
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
            
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            raise
    
    def analyze_question(self, question: str) -> Dict:
        """문제 분석"""
        analysis = {
            "is_multiple_choice": bool(re.search(r'[①②③④⑤]|\b[1-5]\s*[.)]', question)),
            "has_negative": "않" in question or "없" in question or "틀린" in question,
            "keywords": self._extract_keywords(question),
            "complexity": self._estimate_complexity(question)
        }
        return analysis
    
    def _extract_keywords(self, question: str) -> List[str]:
        """키워드 추출"""
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
        """복잡도 추정"""
        if len(question) < 100:
            return "easy"
        elif len(question) < 200:
            return "medium"
        else:
            return "hard"
    
    def create_prompt(self, question: str, analysis: Dict) -> str:
        """프롬프트 생성"""
        if analysis["is_multiple_choice"]:
            return self._create_mc_prompt(question, analysis)
        else:
            return self._create_open_prompt(question, analysis)
    
    def _create_mc_prompt(self, question: str, analysis: Dict) -> str:
        """객관식 프롬프트"""
        system_msg = """당신은 금융보안 전문가입니다. 주어진 객관식 문제를 정확히 분석하고 올바른 답을 선택하세요."""
        
        if analysis["has_negative"]:
            system_msg += " 이 문제는 '틀린 것' 또는 '옳지 않은 것'을 찾는 문제입니다. 주의 깊게 읽어보세요."
        
        prompt = f"""### 지시사항:
{system_msg}

### 문제:
{question}

### 답변 형식:
정답 번호만 출력하세요 (1, 2, 3, 4, 5 중 하나).

### 답변:"""
        
        return prompt
    
    def _create_open_prompt(self, question: str, analysis: Dict) -> str:
        """주관식 프롬프트"""
        domain_context = ""
        if "개인정보" in analysis["keywords"]:
            domain_context = "개인정보보호법과 관련 규정을 고려하여"
        elif "전자금융" in analysis["keywords"]:
            domain_context = "전자금융거래법과 관련 규정을 고려하여"
        elif "보안" in analysis["keywords"]:
            domain_context = "정보보안 관련 법규와 표준을 고려하여"
        
        prompt = f"""### 지시사항:
당신은 금융보안 전문가입니다. {domain_context} 다음 질문에 대해 전문적이고 구체적인 답변을 제공하세요.

### 질문:
{question}

### 답변 가이드:
- 법적 근거가 있다면 명시하세요
- 구체적인 방법이나 절차를 제시하세요
- 실무적인 관점에서 답변하세요

### 답변:"""
        
        return prompt
    
    def generate_response(self, prompt: str, timeout: int = 30) -> str:
        """응답 생성"""
        try:
            # 캐시 확인
            cache_key = hash(prompt)
            if cache_key in self.cache:
                self.stats["cache_hits"] += 1
                return self.cache[cache_key]
            
            # 타임아웃 적용하여 생성
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
                    max_length=2048
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
        """폴백 답변"""
        if "①" in prompt or "1)" in prompt:
            # 객관식인 경우 - 통계적으로 가장 흔한 답
            return "3"
        else:
            # 주관식인 경우
            if "개인정보" in prompt:
                return "개인정보보호법에 따른 안전성 확보조치가 필요하며, 개인정보처리방침을 수립하고 기술적·관리적·물리적 보호조치를 시행해야 합니다."
            elif "전자금융" in prompt:
                return "전자금융거래법에 따른 보안대책을 수립하고, 전자적 전송 및 처리 과정에서의 보안성을 확보해야 합니다."
            else:
                return "금융보안 관련 법규에 따른 체계적인 보안관리체계 구축이 필요합니다."
    
    def extract_answer(self, response: str, is_multiple_choice: bool) -> str:
        """답변 추출"""
        if is_multiple_choice:
            # 숫자 추출
            numbers = re.findall(r'\b([1-5])\b', response)
            if numbers:
                return numbers[0]
            
            # 원 번호 추출
            circle_match = re.search(r'[①②③④⑤]', response)
            if circle_match:
                circle_to_num = {'①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5'}
                return circle_to_num[circle_match.group()]
            
            # 기본값
            return "3"
        else:
            # 주관식 - 응답 정리
            cleaned = response.replace("### 답변:", "").strip()
            if len(cleaned) < 10:  # 너무 짧은 경우
                return self._get_fallback_answer("")
            return cleaned[:500]  # 길이 제한
    
    def cleanup(self):
        """리소스 정리"""
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

class HighPerformanceInferenceEngine:
    """고성능 추론 엔진"""
    
    def __init__(self):
        self.start_time = time.time()
        self.model_handler = OptimizedSOLARInference()
        
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
        
        # 우선순위 정렬 (쉬운 것부터)
        analyzed_questions.sort(key=lambda x: (
            x["analysis"]["complexity"] == "hard",
            x["analysis"]["complexity"] == "medium",
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
            timeout = 60 if analysis["complexity"] == "hard" else 30
            
            # 프롬프트 생성
            prompt = self.model_handler.create_prompt(question, analysis)
            
            # 응답 생성
            response = self.model_handler.generate_response(prompt, timeout)
            
            # 답변 추출
            answer = self.model_handler.extract_answer(response, analysis["is_multiple_choice"])
            predictions[idx] = answer
            
            self.model_handler.stats["total_processed"] += 1
            
            # 메모리 정리 (50개마다)
            if idx % 50 == 0:
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
        """리소스 정리"""
        self.model_handler.cleanup()

def main():
    """메인 함수"""
    
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
        engine = HighPerformanceInferenceEngine()
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