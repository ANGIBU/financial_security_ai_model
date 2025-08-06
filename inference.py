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
import numpy as np
warnings.filterwarnings("ignore")

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Transformers 관련 import
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import logging
logging.set_verbosity_error()

# 커스텀 모듈
from model_handler import ModelHandler
from data_processor import DataProcessor
from prompt_engineering import PromptEngineer
from advanced_optimizer import SystemOptimizer
from pattern_learner import AnswerPatternLearner

class FinancialAIInference:
    """금융 AI 추론 엔진"""
    
    def __init__(self):
        self.start_time = time.time()
        
        # GPU 메모리 설정
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.95)
        
        print("시스템 초기화 중...")
        
        # 컴포넌트 초기화
        self.model_handler = ModelHandler(
            model_name="upstage/SOLAR-10.7B-Instruct-v1.0",
            device="cuda" if torch.cuda.is_available() else "cpu",
            load_in_4bit=False,
            max_memory_gb=22
        )
        
        self.data_processor = DataProcessor()
        self.prompt_engineer = PromptEngineer()
        self.optimizer = SystemOptimizer()
        self.pattern_learner = AnswerPatternLearner()
        
        # 통계
        self.stats = {
            "total": 0,
            "mc_correct": 0,
            "subj_correct": 0,
            "errors": 0,
            "timeouts": 0
        }
        
        print("초기화 완료")
    
    def process_question(self, question: str, question_id: str, idx: int) -> str:
        """단일 문제 처리"""
        
        try:
            # 1. 문제 분석
            structure = self.data_processor.analyze_question_structure(question)
            analysis = self.prompt_engineer.knowledge_base.analyze_question(question)
            
            # 2. 난이도 평가
            difficulty = self.optimizer.evaluate_question_difficulty(question, structure)
            
            # 3. 문제 타입 결정
            is_mc = structure["question_type"] == "multiple_choice"
            
            # 4. 패턴 기반 힌트 생성
            hint_answer, hint_confidence = self.optimizer.get_smart_answer_hint(question, structure)
            
            # 5. 프롬프트 생성
            if is_mc:
                prompt = self._create_mc_prompt(question, structure, analysis, hint_answer, hint_confidence)
            else:
                prompt = self._create_subjective_prompt(question, structure, analysis)
            
            # 6. 모델 추론
            max_attempts = 2 if difficulty.score > 0.7 else 1
            timeout = difficulty.recommended_time
            
            result = self.model_handler.generate_response(
                prompt=prompt,
                question_type=structure["question_type"],
                max_attempts=max_attempts
            )
            
            # 7. 답변 추출 및 후처리
            if is_mc:
                answer = self._extract_mc_answer(result.response, hint_answer, hint_confidence)
            else:
                answer = self._extract_subjective_answer(result.response, structure)
            
            # 8. 패턴 학습
            if is_mc and answer.isdigit():
                self.pattern_learner.update_patterns(question, answer, structure)
            
            return answer
            
        except Exception as e:
            self.stats["errors"] += 1
            
            # 오류 시 폴백
            if structure.get("question_type") == "multiple_choice":
                # 패턴 기반 폴백
                fallback_answer, _ = self.pattern_learner.predict_answer(question, structure)
                return fallback_answer if fallback_answer else "2"
            else:
                return self._get_domain_fallback(structure)
    
    def _create_mc_prompt(self, question: str, structure: Dict, analysis: Dict, 
                         hint_answer: str, hint_confidence: float) -> str:
        """객관식 프롬프트 생성"""
        
        # 부정형 처리
        if structure.get("has_negative", False):
            instruction = "다음 문제에서 해당하지 않는 것 또는 틀린 것을 찾으세요."
        else:
            instruction = "다음 문제의 정답을 선택하세요."
        
        # 도메인 컨텍스트
        context = ""
        if analysis.get("domain"):
            domain = analysis["domain"][0]
            if domain == "개인정보보호":
                context = "개인정보보호법과 관련 규정을 고려하여 답하세요."
            elif domain == "전자금융":
                context = "전자금융거래법과 관련 규정을 고려하여 답하세요."
            elif domain == "정보보안":
                context = "정보보안 관리체계와 보안 원칙을 고려하여 답하세요."
        
        # 힌트 추가 (신뢰도가 높은 경우)
        hint_text = ""
        if hint_confidence > 0.7:
            hint_text = f"\n참고: 이 문제는 {hint_answer}번이 유력한 답변입니다."
        
        prompt = f"""### 지시사항
{instruction}
{context}

### 문제
{question}

### 분석
1. 핵심 개념을 파악합니다.
2. 각 선택지를 검토합니다.
3. 가장 적절한 답을 선택합니다.{hint_text}

### 정답
정답 번호는"""
        
        return prompt
    
    def _create_subjective_prompt(self, question: str, structure: Dict, analysis: Dict) -> str:
        """주관식 프롬프트 생성"""
        
        # 도메인별 지시
        domain_instructions = {
            "개인정보보호": "개인정보보호법에 따른 구체적인 방안을 제시하세요.",
            "전자금융": "전자금융거래법에 따른 안전한 거래 방안을 설명하세요.",
            "정보보안": "정보보안 관리체계 관점에서 체계적으로 답변하세요.",
            "암호화": "암호화 기술과 적용 방안을 구체적으로 설명하세요."
        }
        
        domain = analysis.get("domain", ["일반"])[0]
        instruction = domain_instructions.get(domain, "금융보안 관점에서 구체적으로 답변하세요.")
        
        prompt = f"""### 지시사항
{instruction}

### 질문
{question}

### 답변 구조
1. 핵심 개념 설명
2. 구체적 방안 제시
3. 관련 법령 또는 규정 언급

### 답변
"""
        
        return prompt
    
    def _extract_mc_answer(self, response: str, hint_answer: str, hint_confidence: float) -> str:
        """객관식 답변 추출"""
        
        # 우선순위 패턴
        patterns = [
            (r'정답.*?([1-5])', 1.0),
            (r'답.*?([1-5])', 0.9),
            (r'([1-5])번.*?(?:정답|맞|적절)', 0.8),
            (r'결론.*?([1-5])', 0.7),
            (r'따라서.*?([1-5])', 0.6),
            (r'([1-5])번', 0.5)
        ]
        
        candidates = []
        
        for pattern, weight in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                answer = match.group(1)
                position = match.start() / max(len(response), 1)
                score = weight * (1 - position * 0.3)  # 뒤쪽 답변 선호
                candidates.append((answer, score))
        
        if candidates:
            # 최고 점수 답변
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_answer = candidates[0][0]
            
            # 힌트와 일치하면 보너스
            if best_answer == hint_answer and hint_confidence > 0.6:
                return best_answer
            
            # 신뢰도 차이가 크면 힌트 우선
            if hint_confidence > 0.75 and candidates[0][1] < 0.7:
                return hint_answer
            
            return best_answer
        
        # 숫자만 찾기
        numbers = re.findall(r'[1-5]', response)
        if numbers:
            # 마지막 숫자 반환
            return numbers[-1]
        
        # 힌트 사용
        if hint_confidence > 0.5:
            return hint_answer
        
        # 최종 폴백
        return "2"
    
    def _extract_subjective_answer(self, response: str, structure: Dict) -> str:
        """주관식 답변 추출"""
        
        # 불필요한 접두사 제거
        response = re.sub(r'^(답변|응답|해답)[:：\s]*', '', response, flags=re.IGNORECASE)
        response = response.strip()
        
        # 최소 길이 확인
        if len(response) < 50:
            return self._get_domain_fallback(structure)
        
        # 문장 정리
        sentences = re.split(r'[.!?]\s+', response)
        clean_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                # 중복 제거
                if not clean_sentences or sentence[:20] not in clean_sentences[-1]:
                    clean_sentences.append(sentence)
        
        # 재조립
        result = '. '.join(clean_sentences)
        if result and not result.endswith('.'):
            result += '.'
        
        # 길이 조정
        if len(result) > 1000:
            result = result[:997] + '...'
        elif len(result) < 80:
            # 도메인별 보강
            domain = structure.get("domain", ["일반"])[0]
            if domain == "개인정보보호":
                result += " 개인정보보호법에 따른 추가적인 안전성 확보조치가 필요합니다."
            elif domain == "전자금융":
                result += " 전자금융거래법에 따른 보안대책 수립이 요구됩니다."
        
        return result
    
    def _get_domain_fallback(self, structure: Dict) -> str:
        """도메인별 폴백 답변"""
        domains = structure.get("domain", ["일반"])
        
        if "개인정보보호" in domains:
            return "개인정보보호법에 따라 개인정보의 수집·이용·제공 시 정보주체의 동의를 받아야 하며, 안전성 확보조치를 통해 개인정보를 보호해야 합니다."
        elif "전자금융" in domains:
            return "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보하고, 접근매체를 안전하게 관리하여 이용자를 보호해야 합니다."
        elif "정보보안" in domains:
            return "정보보호관리체계(ISMS)를 구축하여 체계적인 보안 관리와 지속적인 위험 평가를 수행하고, 보안 사고 예방 및 대응 체계를 마련해야 합니다."
        elif "암호화" in domains:
            return "중요 정보는 안전한 암호화 알고리즘을 사용하여 암호화하고, 암호키는 안전하게 관리하며, 전송 구간과 저장 시 모두 암호화를 적용해야 합니다."
        else:
            return "관련 법령과 규정에 따라 적절한 보안 조치를 수립하고, 지속적인 모니터링과 개선을 통해 안전성을 확보해야 합니다."
    
    def execute_inference(self, test_file: str, submission_file: str,
                         output_file: str = "./final_submission.csv") -> Dict:
        """메인 추론 실행"""
        
        # 데이터 로드
        test_df = pd.read_csv(test_file)
        submission_df = pd.read_csv(submission_file)
        
        print(f"데이터 로드 완료: {len(test_df)}개 문항")
        
        # 문제 분석 및 정렬
        print("문제 분석 중...")
        questions_data = []
        
        for idx, row in test_df.iterrows():
            question = row['Question']
            structure = self.data_processor.analyze_question_structure(question)
            difficulty = self.optimizer.evaluate_question_difficulty(question, structure)
            
            questions_data.append({
                "idx": idx,
                "id": row['ID'],
                "question": question,
                "structure": structure,
                "difficulty": difficulty,
                "is_mc": structure["question_type"] == "multiple_choice"
            })
        
        # 최적화된 순서로 정렬 (쉬운 객관식부터)
        questions_data.sort(key=lambda x: (
            not x["is_mc"],  # 객관식 우선
            x["difficulty"].score  # 쉬운 것부터
        ))
        
        mc_count = sum(1 for q in questions_data if q["is_mc"])
        subj_count = len(questions_data) - mc_count
        
        print(f"문제 구성: 객관식 {mc_count}개, 주관식 {subj_count}개")
        
        # 추론 실행
        answers = [""] * len(test_df)
        
        print("추론 시작...")
        for q_data in tqdm(questions_data, desc="추론"):
            idx = q_data["idx"]
            question_id = q_data["id"]
            question = q_data["question"]
            
            # 답변 생성
            answer = self.process_question(question, question_id, idx)
            answers[idx] = answer
            
            self.stats["total"] += 1
            
            # 주기적 메모리 정리
            if self.stats["total"] % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # 결과 저장
        submission_df['Answer'] = answers
        submission_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 통계 계산
        elapsed_time = time.time() - self.start_time
        
        # 답변 분포 분석
        mc_answers = [a for a, q in zip(answers, questions_data) if q["is_mc"] and a.isdigit()]
        answer_distribution = {}
        for ans in mc_answers:
            answer_distribution[ans] = answer_distribution.get(ans, 0) + 1
        
        # 결과 출력
        print("\n" + "="*50)
        print("추론 완료")
        print("="*50)
        print(f"총 문항: {len(answers)}개")
        print(f"소요 시간: {elapsed_time/60:.1f}분")
        print(f"문항당 평균: {elapsed_time/len(answers):.1f}초")
        
        if mc_answers:
            print(f"\n객관식 답변 분포 ({len(mc_answers)}개):")
            for num in sorted(answer_distribution.keys()):
                count = answer_distribution[num]
                pct = (count / len(mc_answers)) * 100
                print(f"  {num}번: {count}개 ({pct:.1f}%)")
        
        print(f"\n결과 파일: {output_file}")
        
        return {
            "success": True,
            "total_questions": len(answers),
            "mc_count": mc_count,
            "subj_count": subj_count,
            "elapsed_minutes": elapsed_time / 60,
            "answer_distribution": answer_distribution
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self.model_handler.cleanup()
            self.data_processor.cleanup()
            self.prompt_engineer.cleanup()
            self.pattern_learner.cleanup()
            
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"정리 중 오류: {e}")

def main():
    """메인 함수"""
    
    print("="*50)
    print("금융 AI Challenge 추론 시스템")
    print("="*50)
    
    # GPU 확인
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_info.name}")
        print(f"메모리: {gpu_info.total_memory / (1024**3):.1f}GB")
    else:
        print("GPU 없음 - CPU 모드")
    
    # 파일 확인
    test_file = "./test.csv"
    submission_file = "./sample_submission.csv"
    
    if not os.path.exists(test_file):
        print(f"오류: {test_file} 파일 없음")
        sys.exit(1)
    
    if not os.path.exists(submission_file):
        print(f"오류: {submission_file} 파일 없음")
        sys.exit(1)
    
    # 추론 실행
    engine = None
    try:
        engine = FinancialAIInference()
        results = engine.execute_inference(test_file, submission_file)
        
        if results["success"]:
            print("\n✅ 추론 완료!")
        
    except KeyboardInterrupt:
        print("\n추론 중단")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if engine:
            engine.cleanup()

if __name__ == "__main__":
    main()