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

# 학습 모듈
from learning_system import UnifiedLearningSystem
from manual_correction import ManualCorrectionSystem
from auto_learner import AutoLearner

class FinancialAIInference:
    """금융 AI 추론 엔진"""
    
    def __init__(self, enable_learning: bool = True):
        self.start_time = time.time()
        self.enable_learning = enable_learning
        
        # GPU 메모리 설정
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.95)
        
        print("시스템 초기화 중...")
        
        # 기존 컴포넌트
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
        
        # 학습 컴포넌트
        if self.enable_learning:
            print("학습 시스템 초기화 중...")
            self.learning_system = UnifiedLearningSystem()
            self.correction_system = ManualCorrectionSystem()
            self.auto_learner = AutoLearner()
            
            # 기존 학습 데이터 로드
            self._load_existing_learning_data()
        
        # 통계
        self.stats = {
            "total": 0,
            "mc_correct": 0,
            "subj_correct": 0,
            "errors": 0,
            "timeouts": 0,
            "learned": 0,
            "korean_failures": 0,
            "korean_fixes": 0
        }
        
        print("초기화 완료")
    
    def _load_existing_learning_data(self) -> None:
        """학습 데이터 로드"""
        
        # 학습 시스템 데이터
        if self.learning_system.load_learning_data():
            print(f"학습 데이터 로드: {self.learning_system.learning_metrics['total_samples']}개")
        
        # 자동 학습 모델
        if self.auto_learner.load_model():
            print(f"자동 학습 모델 로드: {len(self.auto_learner.pattern_weights)}개 패턴")
        
        # 교정 데이터
        corrections = self.correction_system.load_corrections_from_csv("./corrections.csv")
        if corrections > 0:
            print(f"교정 데이터 로드: {corrections}개")
    
    def _validate_korean_answer(self, answer: str, question_type: str) -> bool:
        """한국어 답변 검증 (객관식은 관대하게)"""
        
        # 객관식은 숫자만 있으면 통과
        if question_type == "multiple_choice":
            if answer.isdigit() and 1 <= int(answer) <= 5:
                return True
            # 숫자가 포함되어 있으면 통과
            if re.search(r'[1-5]', answer):
                return True
            return False
        
        # 주관식만 엄격한 검증
        if re.search(r'[\u4e00-\u9fff]', answer):
            return False
        
        korean_chars = len(re.findall(r'[가-힣]', answer))
        total_chars = len(re.sub(r'[^\w]', '', answer))
        
        if total_chars > 0:
            korean_ratio = korean_chars / total_chars
            return korean_ratio >= 0.5
        
        return len(answer.strip()) > 0
    
    def _fix_korean_answer(self, answer: str, question_type: str) -> str:
        """한국어 답변 수정"""
        
        if question_type == "multiple_choice":
            # 객관식에서 숫자 추출
            numbers = re.findall(r'[1-5]', answer)
            if numbers:
                return numbers[-1]
            return "3"  # 기본값
        
        # 주관식 한국어 정리
        fixed = self.data_processor._clean_korean_text(answer)
        
        # 여전히 문제가 있으면 폴백
        if not self._validate_korean_answer(fixed, question_type):
            return self._get_korean_fallback(question_type)
        
        return fixed
    
    def _get_korean_fallback(self, question_type: str) -> str:
        """한국어 폴백 답변"""
        
        if question_type == "multiple_choice":
            return "3"
        else:
            return "관련 법령과 규정에 따른 적절한 조치가 필요합니다."
    
    def process_question(self, question: str, question_id: str, idx: int) -> str:
        """문제 처리"""
        
        try:
            # 1. 문제 분석
            structure = self.data_processor.analyze_question_structure(question)
            analysis = self.prompt_engineer.knowledge_base.analyze_question(question)
            
            # 2. 난이도 평가
            difficulty = self.optimizer.evaluate_question_difficulty(question, structure)
            
            # 3. 문제 타입 결정
            is_mc = structure["question_type"] == "multiple_choice"
            
            # 4. 학습 기반 예측
            if self.enable_learning:
                # 자동 학습 예측
                learned_answer, learned_confidence = self.auto_learner.predict_with_patterns(
                    question, structure["question_type"]
                )
                
                # 교정 시스템 확인
                corrected_answer, correction_conf = self.correction_system.apply_corrections(
                    question, learned_answer
                )
                
                if correction_conf > 0.8:
                    if self._validate_korean_answer(corrected_answer, structure["question_type"]):
                        return corrected_answer
                    else:
                        corrected_answer = self._fix_korean_answer(corrected_answer, structure["question_type"])
                        return corrected_answer
            else:
                learned_answer, learned_confidence = None, 0.0
            
            # 5. 패턴 기반 힌트
            hint_answer, hint_confidence = self.optimizer.get_smart_answer_hint(question, structure)
            
            # 6. 학습과 패턴 결합
            if learned_confidence > hint_confidence:
                final_hint = learned_answer
                final_confidence = learned_confidence
            else:
                final_hint = hint_answer
                final_confidence = hint_confidence
            
            # 7. 프롬프트 생성
            if is_mc:
                prompt = self._create_mc_prompt(
                    question, structure, analysis, final_hint, final_confidence
                )
            else:
                prompt = self._create_subjective_prompt(question, structure, analysis)
            
            # 8. 모델 추론
            max_attempts = 2 if difficulty.score > 0.7 else 1
            result = self.model_handler.generate_response(
                prompt=prompt,
                question_type=structure["question_type"],
                max_attempts=max_attempts
            )
            
            # 9. 답변 추출
            if is_mc:
                answer = self._extract_mc_answer(result.response, final_hint, final_confidence)
            else:
                answer = self._extract_subjective_answer(result.response, structure)
            
            # 10. 한국어 검증 및 수정 (주관식만)
            if not self._validate_korean_answer(answer, structure["question_type"]):
                self.stats["korean_failures"] += 1
                answer = self._fix_korean_answer(answer, structure["question_type"])
                self.stats["korean_fixes"] += 1
            
            # 11. 학습
            if self.enable_learning:
                # 자동 학습
                self.auto_learner.learn_from_prediction(
                    question, answer, result.confidence,
                    structure["question_type"], analysis.get("domain", ["일반"])
                )
                
                # 학습 시스템에 추가
                if result.confidence > 0.6:
                    self.learning_system.add_training_sample(
                        question=question,
                        correct_answer=answer,
                        predicted_answer=answer,
                        confidence=result.confidence,
                        question_type=structure["question_type"],
                        domain=analysis.get("domain", ["일반"]),
                        question_id=question_id
                    )
                    self.stats["learned"] += 1
            
            # 12. 패턴 학습
            if is_mc and answer.isdigit():
                self.pattern_learner.update_patterns(question, answer, structure)
            
            return answer
            
        except Exception as e:
            self.stats["errors"] += 1
            
            # 학습 기반 폴백
            if self.enable_learning:
                fallback, conf = self.auto_learner.predict_with_patterns(
                    question, structure.get("question_type", "multiple_choice")
                )
                if conf > 0.5 and self._validate_korean_answer(fallback, structure.get("question_type")):
                    return fallback
            
            # 기존 폴백
            if structure.get("question_type") == "multiple_choice":
                fallback_answer, _ = self.pattern_learner.predict_answer(question, structure)
                return fallback_answer if fallback_answer else "3"
            else:
                return self._get_korean_fallback(structure.get("question_type", "subjective"))
    
    def _create_mc_prompt(self, question: str, structure: Dict, 
                         analysis: Dict, hint_answer: str, 
                         hint_confidence: float) -> str:
        """객관식 프롬프트 생성 (간단하게)"""
        
        # 부정형 처리
        if structure.get("has_negative", False):
            instruction = "다음 문제에서 해당하지 않는 것 또는 틀린 것을 찾으세요."
        else:
            instruction = "다음 문제의 정답을 선택하세요."
        
        # 학습 기반 힌트
        hint_text = ""
        if hint_confidence > 0.75:
            hint_text = f"\n참고: 유사 문제 분석 결과 {hint_answer}번이 유력합니다."
        
        prompt = f"""### 지시사항
당신은 금융보안 전문가입니다.
{instruction}

### 문제
{question}

간단히 분석한 후 정답 번호만 제시하세요.{hint_text}

정답:"""
        
        return prompt
    
    def _create_subjective_prompt(self, question: str, structure: Dict, 
                                 analysis: Dict) -> str:
        """주관식 프롬프트 생성"""
        
        # 도메인별 지시
        domain_instructions = {
            "개인정보보호": "개인정보보호법에 따른 구체적인 방안을 제시하세요.",
            "전자금융": "전자금융거래법에 따른 안전한 거래 방안을 설명하세요.",
            "정보보안": "정보보안 관리체계 관점에서 체계적으로 답변하세요."
        }
        
        domain = analysis.get("domain", ["일반"])[0]
        instruction = domain_instructions.get(domain, "금융보안 관점에서 구체적으로 답변하세요.")
        
        prompt = f"""### 지시사항
당신은 금융보안 전문가입니다.
반드시 한국어로만 답변하세요.

{instruction}

### 질문
{question}

### 답변 구조
1. 핵심 개념 설명
2. 구체적 방안 제시
3. 관련 법령 또는 규정 언급

답변:"""
        
        return prompt
    
    def _extract_mc_answer(self, response: str, hint_answer: str, 
                          hint_confidence: float) -> str:
        """객관식 답변 추출"""
        
        # 응답 정리
        cleaned_response = response.strip()
        
        # 단순 숫자 확인
        if re.match(r'^[1-5]$', cleaned_response):
            return cleaned_response
        
        # 패턴 기반 추출
        patterns = [
            (r'정답.*?([1-5])', 1.0),
            (r'답.*?([1-5])', 0.9),
            (r'([1-5])번.*?(?:정답|맞|적절)', 0.8),
            (r'결론.*?([1-5])', 0.7),
            (r'따라서.*?([1-5])', 0.6),
            (r'([1-5])번', 0.5),
            (r'([1-5])', 0.3)
        ]
        
        candidates = []
        
        for pattern, weight in patterns:
            matches = re.finditer(pattern, cleaned_response, re.IGNORECASE)
            for match in matches:
                answer = match.group(1)
                position = match.start() / max(len(cleaned_response), 1)
                score = weight * (1 - position * 0.3)
                candidates.append((answer, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_answer = candidates[0][0]
            
            # 학습된 힌트와 일치 시 보너스
            if best_answer == hint_answer and hint_confidence > 0.7:
                return best_answer
            
            return best_answer
        
        # 힌트 사용
        if hint_confidence > 0.6:
            return hint_answer
        
        return "3"
    
    def _extract_subjective_answer(self, response: str, structure: Dict) -> str:
        """주관식 답변 추출"""
        
        # 한국어 정리
        cleaned_response = self.data_processor._clean_korean_text(response)
        
        # 불필요한 접두사 제거
        cleaned_response = re.sub(r'^(답변|응답|해답)[:：\s]*', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = cleaned_response.strip()
        
        # 최소 길이 확인
        if len(cleaned_response) < 50:
            return self._get_domain_fallback_korean(structure)
        
        # 문장 정리
        sentences = re.split(r'[.!?]\s+', cleaned_response)
        clean_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
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
            domain = structure.get("domain_hints", ["일반"])[0] if structure.get("domain_hints") else "일반"
            if domain == "개인정보보호":
                result += " 개인정보보호법에 따른 추가적인 안전성 확보조치가 필요합니다."
            elif domain == "전자금융":
                result += " 전자금융거래법에 따른 보안대책 수립이 요구됩니다."
        
        return result
    
    def _get_domain_fallback_korean(self, structure: Dict) -> str:
        """도메인별 한국어 폴백 답변"""
        domains = structure.get("domain_hints", ["일반"])
        
        if "개인정보보호" in domains:
            return "개인정보보호법에 따라 개인정보의 수집과 이용, 제공 시 정보주체의 동의를 받아야 하며, 안전성 확보조치를 통해 개인정보를 보호해야 합니다."
        elif "전자금융" in domains:
            return "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보하고, 접근매체를 안전하게 관리하여 이용자를 보호해야 합니다."
        else:
            return "관련 법령과 규정에 따라 적절한 보안 조치를 수립하고, 지속적인 모니터링과 개선을 통해 안전성을 확보해야 합니다."
    
    def execute_inference(self, test_file: str, submission_file: str,
                         output_file: str = "./final_submission.csv",
                         enable_manual_correction: bool = False) -> Dict:
        """추론 실행"""
        
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
        
        # 최적화된 순서로 정렬
        questions_data.sort(key=lambda x: (
            not x["is_mc"],
            x["difficulty"].score
        ))
        
        mc_count = sum(1 for q in questions_data if q["is_mc"])
        subj_count = len(questions_data) - mc_count
        
        print(f"문제 구성: 객관식 {mc_count}개, 주관식 {subj_count}개")
        
        if self.enable_learning:
            print(f"학습 모드: 활성화")
        
        # 추론 실행
        answers = [""] * len(test_df)
        predictions = []
        
        print("추론 시작...")
        for q_data in tqdm(questions_data, desc="추론"):
            idx = q_data["idx"]
            question_id = q_data["id"]
            question = q_data["question"]
            
            # 답변 생성
            answer = self.process_question(question, question_id, idx)
            answers[idx] = answer
            predictions.append({"question": question, "answer": answer, "id": question_id})
            
            self.stats["total"] += 1
            
            # 주기적 메모리 정리
            if self.stats["total"] % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            # 학습 최적화 (50문제마다)
            if self.enable_learning and self.stats["total"] % 50 == 0:
                self.learning_system.optimize_rules()
                self.auto_learner.optimize_patterns()
        
        # 수동 교정 (선택사항)
        if enable_manual_correction and self.enable_learning:
            print("\n수동 교정 모드 시작...")
            corrections = self.correction_system.interactive_correction(
                questions_data[:10],  # 처음 10개만
                answers[:10]
            )
            print(f"교정 완료: {corrections}개")
        
        # 결과 저장
        submission_df['Answer'] = answers
        submission_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 학습 데이터 저장
        if self.enable_learning:
            self.learning_system.save_learning_data()
            self.auto_learner.save_model()
            self.correction_system.save_corrections_to_csv()
        
        # 통계 계산
        elapsed_time = time.time() - self.start_time
        
        # 답변 분포 분석
        mc_answers = [a for a, q in zip(answers, questions_data) if q["is_mc"] and a.isdigit()]
        answer_distribution = {}
        for ans in mc_answers:
            answer_distribution[ans] = answer_distribution.get(ans, 0) + 1
        
        # 한국어 품질 검사 (주관식만)
        subj_answers = [a for a, q in zip(answers, questions_data) if not q["is_mc"]]
        korean_quality_issues = 0
        total_korean_ratio = 0
        
        for answer in subj_answers:
            korean_chars = len(re.findall(r'[가-힣]', answer))
            total_chars = len(re.sub(r'[^\w]', '', answer))
            
            if total_chars > 0:
                korean_ratio = korean_chars / total_chars
                total_korean_ratio += korean_ratio
                
                if korean_ratio < 0.5 or re.search(r'[\u4e00-\u9fff]', answer):
                    korean_quality_issues += 1
        
        avg_korean_ratio = total_korean_ratio / max(len(subj_answers), 1) if subj_answers else 1.0
        
        # 결과 출력
        print("\n" + "="*50)
        print("추론 완료")
        print("="*50)
        print(f"총 문항: {len(answers)}개")
        print(f"소요 시간: {elapsed_time/60:.1f}분")
        print(f"문항당 평균: {elapsed_time/len(answers):.1f}초")
        
        # 한국어 품질 리포트
        print(f"\n한국어 품질 리포트:")
        print(f"  한국어 수정: {self.stats['korean_fixes']}회")
        print(f"  품질 문제: {korean_quality_issues}/{len(subj_answers)}개 (주관식)")
        print(f"  평균 한국어 비율: {avg_korean_ratio:.2%}")
        if korean_quality_issues == 0:
            print("  ✅ 모든 답변이 한국어로 생성됨")
        
        if self.enable_learning:
            print(f"\n학습 통계:")
            print(f"  학습된 샘플: {self.stats['learned']}개")
            print(f"  패턴 수: {len(self.auto_learner.pattern_weights)}개")
            print(f"  정확도: {self.learning_system.get_current_accuracy():.2%}")
        
        if mc_answers:
            print(f"\n객관식 답변 분포 ({len(mc_answers)}개):")
            for num in sorted(answer_distribution.keys()):
                count = answer_distribution[num]
                pct = (count / len(mc_answers)) * 100
                print(f"  {num}번: {count}개 ({pct:.1f}%)")
            
            # 다양성 확인
            unique_answers = len(answer_distribution)
            if unique_answers >= 4:
                print("  ✅ 다양한 답변 생성됨 (폴백 현상 해결)")
            else:
                print("  ⚠️ 답변 다양성 부족")
        
        print(f"\n결과 파일: {output_file}")
        
        return {
            "success": True,
            "total_questions": len(answers),
            "mc_count": mc_count,
            "subj_count": subj_count,
            "elapsed_minutes": elapsed_time / 60,
            "answer_distribution": answer_distribution,
            "korean_quality": {
                "fixes": self.stats["korean_fixes"],
                "issues": korean_quality_issues,
                "avg_ratio": avg_korean_ratio,
                "success_rate": (len(subj_answers) - korean_quality_issues) / max(len(subj_answers), 1)
            },
            "learning_stats": {
                "learned_samples": self.stats["learned"],
                "patterns": len(self.auto_learner.pattern_weights) if self.enable_learning else 0,
                "accuracy": self.learning_system.get_current_accuracy() if self.enable_learning else 0
            }
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self.model_handler.cleanup()
            self.data_processor.cleanup()
            self.prompt_engineer.cleanup()
            self.pattern_learner.cleanup()
            
            if self.enable_learning:
                self.learning_system.cleanup()
                self.correction_system.cleanup()
                self.auto_learner.cleanup()
            
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
    
    # 학습 모드 선택
    response = input("\n학습 기능을 활성화하시겠습니까? (y/n): ")
    enable_learning = response.lower() == 'y'
    
    # 추론 실행
    engine = None
    try:
        engine = FinancialAIInference(enable_learning=enable_learning)
        results = engine.execute_inference(
            test_file, 
            submission_file,
            enable_manual_correction=False
        )
        
        if results["success"]:
            print("\n✅ 추론 완료!")
            if results["korean_quality"]["success_rate"] > 0.9:
                print("✅ 한국어 품질 우수!")
            else:
                print(f"⚠️ 한국어 품질: {results['korean_quality']['success_rate']:.2%}")
        
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