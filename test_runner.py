# test_runner.py

"""
테스트 실행기
- 50문항 테스트 실행
- 빠른 성능 검증
- 간단한 결과 분석
"""

import os
import sys
import time
import pandas as pd
import torch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from model_handler import ModelHandler
from data_processor import DataProcessor
from prompt_engineering import PromptEngineer
from learning_system import LearningSystem

class TestRunner:
    
    def __init__(self, test_size: int = 50):
        self.test_size = test_size
        self.start_time = time.time()
        
        print(f"테스트 실행기 초기화 중... (대상: {test_size}문항)")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.model_handler = ModelHandler(
            model_name="upstage/SOLAR-10.7B-Instruct-v1.0",
            device="cuda" if torch.cuda.is_available() else "cpu",
            load_in_4bit=True,
            max_memory_gb=22,
            verbose=False
        )
        
        self.data_processor = DataProcessor(debug_mode=False)
        self.prompt_engineer = PromptEngineer()
        self.learning_system = LearningSystem(debug_mode=False)
        
        try:
            self.learning_system.load_model()
            print("기존 학습 데이터 로드 완료")
        except:
            print("새로운 학습 세션 시작")
        
        self.stats = {
            "total": 0,
            "mc_count": 0,
            "subj_count": 0,
            "model_success": 0,
            "pattern_success": 0,
            "fallback_used": 0,
            "korean_quality_sum": 0.0,
            "processing_times": []
        }
        
        print("초기화 완료\n")
    
    def load_test_data(self, test_file: str, submission_file: str) -> tuple:
        if not os.path.exists(test_file):
            print(f"오류: {test_file} 파일을 찾을 수 없습니다")
            return None, None
        
        if not os.path.exists(submission_file):
            print(f"오류: {submission_file} 파일을 찾을 수 없습니다")
            return None, None
        
        test_df = pd.read_csv(test_file)
        submission_df = pd.read_csv(submission_file)
        
        if len(test_df) < self.test_size:
            print(f"경고: 전체 {len(test_df)}문항, 요청 {self.test_size}문항")
            self.test_size = len(test_df)
        
        test_sample = test_df.head(self.test_size).copy()
        submission_sample = submission_df.head(self.test_size).copy()
        
        print(f"테스트 데이터 로드: {len(test_sample)}문항")
        return test_sample, submission_sample
    
    def analyze_questions(self, test_df: pd.DataFrame) -> dict:
        mc_count = 0
        subj_count = 0
        
        for _, row in test_df.iterrows():
            question = row['Question']
            structure = self.data_processor.analyze_question_structure(question)
            
            if structure["question_type"] == "multiple_choice":
                mc_count += 1
            else:
                subj_count += 1
        
        print(f"문제 구성: 객관식 {mc_count}개, 주관식 {subj_count}개")
        
        return {
            "mc_count": mc_count,
            "subj_count": subj_count,
            "total": mc_count + subj_count
        }
    
    def process_single_question(self, question: str, question_id: str, idx: int) -> str:
        start_time = time.time()
        
        try:
            structure = self.data_processor.analyze_question_structure(question)
            is_mc = structure["question_type"] == "multiple_choice"
            
            if is_mc:
                self.stats["mc_count"] += 1
                
                hint_answer, hint_confidence = self.learning_system.get_smart_answer_hint(question, structure)
                
                if hint_confidence > 0.6:
                    self.stats["pattern_success"] += 1
                    answer = hint_answer
                else:
                    prompt = self.prompt_engineer.create_korean_reinforced_prompt(question, "multiple_choice")
                    
                    result = self.model_handler.generate_response(
                        prompt=prompt,
                        question_type="multiple_choice",
                        max_attempts=1
                    )
                    
                    extracted = self.data_processor.extract_mc_answer_fast(result.response)
                    
                    if extracted and extracted.isdigit() and 1 <= int(extracted) <= 5:
                        self.stats["model_success"] += 1
                        answer = extracted
                    else:
                        self.stats["fallback_used"] += 1
                        answer = str((hash(question) % 5) + 1)
            
            else:
                self.stats["subj_count"] += 1
                
                prompt = self.prompt_engineer.create_korean_reinforced_prompt(question, "subjective")
                
                result = self.model_handler.generate_response(
                    prompt=prompt,
                    question_type="subjective",
                    max_attempts=1
                )
                
                answer = self.data_processor._clean_korean_text(result.response)
                
                is_valid, quality = self._validate_korean_quality(answer)
                self.stats["korean_quality_sum"] += quality
                
                if not is_valid or quality < 0.4:
                    self.stats["fallback_used"] += 1
                    answer = self._get_fallback_answer(question)
                else:
                    self.stats["model_success"] += 1
                
                if len(answer) < 25:
                    answer = self._get_fallback_answer(question)
                elif len(answer) > 800:
                    answer = answer[:797] + "..."
            
            self.stats["total"] += 1
            processing_time = time.time() - start_time
            self.stats["processing_times"].append(processing_time)
            
            if self.stats["total"] % 10 == 0:
                print(f"  진행: {self.stats['total']}/{self.test_size} ({self.stats['total']/self.test_size*100:.0f}%)")
            
            return answer
            
        except Exception as e:
            print(f"  오류 발생 (문항 {idx}): {str(e)[:50]}")
            self.stats["fallback_used"] += 1
            return "2" if is_mc else "관련 규정에 따라 적절한 조치를 수행해야 합니다."
    
    def _validate_korean_quality(self, text: str) -> tuple:
        if not text or len(text) < 10:
            return False, 0.0
        
        korean_chars = len([c for c in text if '가' <= c <= '힣'])
        total_chars = len([c for c in text if c.isalnum()])
        
        if total_chars == 0:
            return False, 0.0
        
        korean_ratio = korean_chars / total_chars
        
        if korean_ratio < 0.3:
            return False, korean_ratio
        
        if any(ord(c) >= 0x4e00 and ord(c) <= 0x9fff for c in text):
            return False, 0.1
        
        return True, korean_ratio
    
    def _get_fallback_answer(self, question: str) -> str:
        question_lower = question.lower()
        
        if "트로이" in question_lower or "악성코드" in question_lower:
            return "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결과 시스템 리소스 사용 증가가 있습니다."
        elif "개인정보" in question_lower:
            return "개인정보보호법에 따라 개인정보의 안전한 관리와 정보주체의 권리 보호를 위한 체계적인 조치가 필요합니다."
        elif "전자금융" in question_lower:
            return "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보하고 이용자를 보호해야 합니다."
        else:
            return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 개선을 수행해야 합니다."
    
    def run_test(self, test_file: str = "./test.csv", submission_file: str = "./sample_submission.csv"):
        print("="*50)
        print(f"테스트 실행 시작 ({self.test_size}문항)")
        print("="*50)
        
        test_df, submission_df = self.load_test_data(test_file, submission_file)
        
        if test_df is None or submission_df is None:
            return
        
        question_analysis = self.analyze_questions(test_df)
        
        print(f"\n추론 시작...")
        
        answers = []
        
        for idx, row in test_df.iterrows():
            question = row['Question']
            question_id = row['ID']
            
            answer = self.process_single_question(question, question_id, idx)
            answers.append(answer)
        
        submission_df['Answer'] = answers
        
        output_file = f"./test_result_{self.test_size}.csv"
        submission_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        self._print_results(output_file, question_analysis)
        
        try:
            self.learning_system.save_model()
            print("학습 데이터 저장 완료")
        except:
            print("학습 데이터 저장 실패")
    
    def _print_results(self, output_file: str, question_analysis: dict):
        total_time = time.time() - self.start_time
        avg_time = sum(self.stats["processing_times"]) / len(self.stats["processing_times"])
        
        print(f"\n" + "="*50)
        print("테스트 완료")
        print("="*50)
        
        print(f"처리 시간: {total_time:.1f}초")
        print(f"문항당 평균: {avg_time:.2f}초")
        
        print(f"\n처리 통계:")
        success_rate = self.stats["model_success"] / self.stats["total"] * 100
        pattern_rate = self.stats["pattern_success"] / self.stats["total"] * 100
        fallback_rate = self.stats["fallback_used"] / self.stats["total"] * 100
        
        print(f"  모델 생성 성공: {self.stats['model_success']}/{self.stats['total']} ({success_rate:.1f}%)")
        print(f"  패턴 매칭 성공: {self.stats['pattern_success']}/{self.stats['total']} ({pattern_rate:.1f}%)")
        print(f"  폴백 사용: {self.stats['fallback_used']}/{self.stats['total']} ({fallback_rate:.1f}%)")
        
        if self.stats["subj_count"] > 0:
            avg_korean_quality = self.stats["korean_quality_sum"] / self.stats["subj_count"]
            print(f"  평균 한국어 품질: {avg_korean_quality:.2f}")
        
        print(f"\n결과 파일: {output_file}")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"최대 GPU 메모리 사용: {memory_used:.1f}GB")
    
    def cleanup(self):
        try:
            self.model_handler.cleanup()
            self.data_processor.cleanup()
            self.prompt_engineer.cleanup()
            self.learning_system.cleanup()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("정리 완료")
        except Exception as e:
            print(f"정리 중 오류: {e}")

def main():
    test_size = 50
    
    if len(sys.argv) > 1:
        try:
            test_size = int(sys.argv[1])
            test_size = max(1, min(test_size, 500))
        except:
            print("잘못된 문항 수, 기본값 50 사용")
            test_size = 50
    
    print(f"테스트 실행기 시작 (Python {sys.version.split()[0]})")
    
    runner = None
    try:
        runner = TestRunner(test_size=test_size)
        runner.run_test()
        
    except KeyboardInterrupt:
        print("\n테스트 중단")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if runner:
            runner.cleanup()

if __name__ == "__main__":
    main()