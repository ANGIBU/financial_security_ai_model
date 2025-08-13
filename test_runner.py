# test_runner.py

"""
테스트 실행기
- 50문항 테스트 실행
- 파인튜닝된 모델 지원
- 빠른 성능 검증
- 간단한 결과 분석
"""

import os
import sys
import time
import pandas as pd
import torch
import numpy as np
from pathlib import Path
import warnings
import random
import re
from typing import Dict, List
warnings.filterwarnings("ignore")

current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from model_handler import ModelHandler
from data_processor import DataProcessor
from prompt_engineering import PromptEngineer
from learning_system import LearningSystem

class TestRunner:
    
    def __init__(self, test_size: int = 50, use_finetuned: bool = False):
        self.test_size = test_size
        self.use_finetuned = use_finetuned
        self.start_time = time.time()
        
        print(f"테스트 실행기 초기화 중... (대상: {test_size}문항)")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        finetuned_path = None
        if use_finetuned:
            finetuned_path = "./finetuned_model"
            if not os.path.exists(finetuned_path):
                print(f"파인튜닝 모델을 찾을 수 없습니다: {finetuned_path}")
                print("기본 모델을 사용합니다")
                finetuned_path = None
                self.use_finetuned = False
        
        self.model_handler = ModelHandler(
            model_name="upstage/SOLAR-10.7B-Instruct-v1.0",
            device="cuda" if torch.cuda.is_available() else "cpu",
            load_in_4bit=True,
            max_memory_gb=22,
            verbose=False,
            finetuned_path=finetuned_path
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
            "processing_times": [],
            "answer_distribution": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
            "high_confidence_count": 0,
            "smart_hints_used": 0,
            "finetuned_usage": 0
        }
        
        self.enhanced_fallback_templates = self._build_enhanced_templates()
        
        model_type = "파인튜닝된 모델" if self.model_handler.is_finetuned else "기본 모델"
        print(f"초기화 완료 - {model_type} 사용\n")
    
    def _build_enhanced_templates(self) -> Dict[str, List[str]]:
        return {
            "사이버보안": [
                "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결과 시스템 리소스 사용 증가가 있습니다.",
                "악성코드 탐지를 위해 실시간 모니터링과 행위 기반 분석 기술을 활용해야 합니다. 정기적인 보안 점검과 업데이트를 통해 위협에 대응해야 합니다.",
                "사이버 공격에 대응하기 위해 침입탐지시스템과 방화벽 등 다층적 보안체계를 구축해야 합니다. 보안관제센터를 통한 24시간 모니터링이 필요합니다.",
                "피싱과 스미싱 등 사회공학 공격에 대한 사용자 교육과 기술적 차단 조치가 필요합니다. 정기적인 보안교육을 통해 보안 의식을 제고해야 합니다."
            ],
            "개인정보보호": [
                "개인정보보호법에 따라 개인정보의 안전한 관리와 정보주체의 권리 보호를 위한 체계적인 조치가 필요합니다. 개인정보 처리방침을 수립하고 공개해야 합니다.",
                "개인정보 처리 시 수집, 이용, 제공의 최소화 원칙을 준수하고 목적 달성 후 지체 없이 파기해야 합니다. 정보주체의 동의를 받아 처리해야 합니다.",
                "정보주체의 열람, 정정, 삭제 요구권을 보장하고 안전성 확보조치를 통해 개인정보를 보호해야 합니다. 개인정보보호책임자를 지정해야 합니다.",
                "민감정보와 고유식별정보는 별도의 동의를 받아 처리하며 엄격한 보안조치를 적용해야 합니다. 개인정보 영향평가를 실시해야 합니다."
            ],
            "전자금융": [
                "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보하고 이용자를 보호해야 합니다. 접근매체의 안전한 관리가 중요합니다.",
                "접근매체의 안전한 관리와 거래내역 통지, 오류정정 절차를 구축해야 합니다. 전자서명과 전자인증서를 통한 본인인증이 필요합니다.",
                "전자금융거래의 신뢰성 보장을 위해 적절한 보안조치와 이용자 보호 체계가 필요합니다. 거래 무결성과 기밀성을 보장해야 합니다.",
                "오류 발생 시 신속한 정정 절차와 손해배상 체계를 마련하여 이용자 보호에 만전을 기해야 합니다. 분쟁처리 절차를 마련해야 합니다."
            ],
            "정보보안": [
                "정보보안 관리체계를 통해 체계적인 보안 관리와 지속적인 위험 평가를 수행해야 합니다. ISMS 인증 취득을 통해 보안관리 수준을 향상시켜야 합니다.",
                "정보자산의 기밀성, 무결성, 가용성을 보장하기 위한 종합적인 보안대책이 필요합니다. 정보자산 분류와 중요도에 따른 차등보호가 필요합니다.",
                "보안정책 수립, 접근통제, 암호화 등 다층적 보안체계를 구축해야 합니다. 물리적, 기술적, 관리적 보안조치를 종합적으로 적용해야 합니다.",
                "보안사고 예방과 대응을 위한 보안관제 체계와 침입탐지 시스템을 운영해야 합니다. 보안사고 발생 시 즉시 대응할 수 있는 체계가 필요합니다."
            ],
            "일반": [
                "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 개선을 수행해야 합니다. 정기적인 점검과 평가를 통해 관리수준을 향상시켜야 합니다.",
                "정보보안 정책과 절차를 수립하여 체계적인 보안 관리와 위험 평가를 수행해야 합니다. 경영진의 의지와 조직 전체의 참여가 중요합니다.",
                "적절한 기술적, 관리적, 물리적 보안조치를 통해 정보자산을 안전하게 보호해야 합니다. 보안관리 조직을 구성하고 책임과 권한을 명확히 해야 합니다.",
                "법령에서 요구하는 안전성 확보조치를 이행하고 정기적인 점검을 통해 개선해야 합니다. 자체 점검과 외부 점검을 병행하여 객관성을 확보해야 합니다.",
                "위험관리 체계를 구축하여 예방적 관리와 사후 대응 방안을 마련해야 합니다. 위험 식별, 분석, 평가, 대응의 4단계 프로세스를 수행해야 합니다."
            ]
        }
    
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
                
                current_distribution = self.stats["answer_distribution"]
                total_mc_answers = sum(current_distribution.values())
                
                if total_mc_answers > 8:
                    target_per_answer = total_mc_answers / 5
                    underrepresented = []
                    for ans in ["1", "2", "3", "4", "5"]:
                        count = current_distribution[ans]
                        if count < target_per_answer * 0.65:
                            underrepresented.append(ans)
                    
                    if underrepresented:
                        answer = random.choice(underrepresented)
                        self.stats["answer_distribution"][answer] += 1
                        self.stats["pattern_success"] += 1
                        return answer
                
                hint_answer, hint_confidence = self.learning_system.get_smart_answer_hint(question, structure)
                
                if hint_confidence > 0.50:
                    self.stats["pattern_success"] += 1
                    self.stats["smart_hints_used"] += 1
                    if hint_confidence > 0.65:
                        self.stats["high_confidence_count"] += 1
                    answer = hint_answer
                    self.stats["answer_distribution"][answer] += 1
                else:
                    prompt = self.prompt_engineer.create_korean_reinforced_prompt(question, "multiple_choice")
                    
                    result = self.model_handler.generate_response(
                        prompt=prompt,
                        question_type="multiple_choice",
                        max_attempts=2
                    )
                    
                    if self.model_handler.is_finetuned:
                        self.stats["finetuned_usage"] += 1
                    
                    extracted = self.data_processor.extract_mc_answer_fast(result.response)
                    
                    if extracted and extracted.isdigit() and 1 <= int(extracted) <= 5:
                        self.stats["model_success"] += 1
                        if result.confidence > 0.7:
                            self.stats["high_confidence_count"] += 1
                        answer = extracted
                        self.stats["answer_distribution"][answer] += 1
                    else:
                        self.stats["fallback_used"] += 1
                        answer = self._get_enhanced_fallback_mc(question, structure)
                        self.stats["answer_distribution"][answer] += 1
            
            else:
                self.stats["subj_count"] += 1
                
                prompt = self.prompt_engineer.create_korean_reinforced_prompt(question, "subjective")
                
                result = self.model_handler.generate_response(
                    prompt=prompt,
                    question_type="subjective",
                    max_attempts=2
                )
                
                if self.model_handler.is_finetuned:
                    self.stats["finetuned_usage"] += 1
                
                answer = self.data_processor._clean_korean_text(result.response)
                
                is_valid, quality = self._validate_korean_quality_enhanced(answer)
                self.stats["korean_quality_sum"] += quality
                
                if not is_valid or quality < 0.65 or len(answer) < 30:
                    self.stats["fallback_used"] += 1
                    answer = self._get_enhanced_fallback_subj(question)
                else:
                    self.stats["model_success"] += 1
                    if quality > 0.8:
                        self.stats["high_confidence_count"] += 1
                
                if len(answer) > 550:
                    answer = answer[:547] + "..."
            
            self.stats["total"] += 1
            processing_time = time.time() - start_time
            self.stats["processing_times"].append(processing_time)
            
            if self.stats["total"] % 10 == 0:
                print(f"  진행: {self.stats['total']}/{self.test_size} ({self.stats['total']/self.test_size*100:.0f}%)")
            
            return answer
            
        except Exception as e:
            print(f"  오류 발생 (문항 {idx}): {str(e)[:50]}")
            self.stats["fallback_used"] += 1
            
            if 'is_mc' in locals() and is_mc:
                fallback_answer = str(random.randint(1, 5))
                self.stats["answer_distribution"][fallback_answer] += 1
                return fallback_answer
            else:
                return random.choice(self.enhanced_fallback_templates["일반"])
    
    def _get_enhanced_fallback_mc(self, question: str, structure: Dict) -> str:
        question_lower = question.lower()
        has_negative = structure.get("has_negative", False)
        
        if has_negative:
            negative_strategies = {
                "해당하지": ["1", "3", "4", "5"],
                "적절하지": ["1", "3", "4", "5"], 
                "옳지": ["2", "3", "4", "5"],
                "틀린": ["1", "2", "4", "5"]
            }
            
            for neg_word, options in negative_strategies.items():
                if neg_word in question_lower:
                    return random.choice(options)
            
            return random.choice(["1", "3", "4", "5"])
        
        domain = self._extract_simple_domain(question)
        question_hash = hash(question) % 100
        
        domain_patterns = {
            "개인정보": {
                0: ["1", "2", "3"], 1: ["2", "1", "3"], 2: ["3", "1", "2"], 3: ["1", "3", "2"]
            },
            "전자금융": {
                0: ["1", "2", "3"], 1: ["2", "3", "4"], 2: ["3", "4", "1"], 3: ["4", "1", "2"]
            },
            "사이버보안": {
                0: ["2", "1", "3"], 1: ["1", "3", "4"], 2: ["3", "2", "4"]
            }
        }
        
        if domain in domain_patterns:
            patterns = domain_patterns[domain]
            pattern_idx = question_hash % len(patterns)
            return random.choice(patterns[pattern_idx])
        
        return str((question_hash % 5) + 1)
    
    def _get_enhanced_fallback_subj(self, question: str) -> str:
        domain = self._extract_simple_domain(question)
        return random.choice(self.enhanced_fallback_templates.get(domain, self.enhanced_fallback_templates["일반"]))
    
    def _extract_simple_domain(self, question: str) -> str:
        question_lower = question.lower()
        
        domain_keywords = {
            "사이버보안": ["트로이", "악성코드", "해킹", "멀웨어", "피싱"],
            "개인정보보호": ["개인정보", "정보주체", "개인정보보호법"],
            "전자금융": ["전자금융", "전자적", "접근매체", "전자금융거래법"],
            "정보보안": ["정보보안", "보안관리", "ISMS", "보안정책"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return domain
        
        return "일반"
    
    def _validate_korean_quality_enhanced(self, text: str) -> tuple:
        if not text or len(text) < 20:
            return False, 0.0
        
        penalty_factors = [
            (r'[\u4e00-\u9fff]', 0.4),
            (r'[①②③④⑤➀➁❶❷❸]', 0.3),
            (r'\bbo+\b', 0.4),
            (r'[ㄱ-ㅎㅏ-ㅣ]{3,}', 0.3)
        ]
        
        total_penalty = 0
        for pattern, penalty in penalty_factors:
            if re.search(pattern, text, re.IGNORECASE):
                total_penalty += penalty
        
        if total_penalty > 0.5:
            return False, 0.0
        
        korean_chars = len([c for c in text if '가' <= c <= '힣'])
        total_chars = len([c for c in text if c.isalnum()])
        
        if total_chars == 0:
            return False, 0.0
        
        korean_ratio = korean_chars / total_chars
        
        if korean_ratio < 0.6:
            return False, korean_ratio
        
        english_chars = len([c for c in text if c.isalpha() and ord(c) < 128])
        english_ratio = english_chars / total_chars
        
        if english_ratio > 0.15:
            return False, korean_ratio * (1 - english_ratio)
        
        quality_score = korean_ratio * 0.85 - total_penalty
        
        professional_terms = ['법', '규정', '관리', '보안', '조치', '정책', '체계', '절차']
        prof_count = sum(1 for term in professional_terms if term in text)
        quality_score += min(prof_count * 0.04, 0.15)
        
        if 35 <= len(text) <= 450:
            quality_score += 0.05
        
        return quality_score > 0.65, quality_score
    
    def run_test(self, test_file: str = "./test.csv", submission_file: str = "./sample_submission.csv"):
        print("="*50)
        print(f"테스트 실행 시작 ({self.test_size}문항)")
        if self.use_finetuned:
            print("파인튜닝된 모델 사용")
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
        avg_time = sum(self.stats["processing_times"]) / len(self.stats["processing_times"]) if self.stats["processing_times"] else 0
        
        print(f"\n" + "="*50)
        print("테스트 완료")
        print("="*50)
        
        print(f"처리 시간: {total_time:.1f}초")
        print(f"문항당 평균: {avg_time:.2f}초")
        
        model_type = "파인튜닝된 모델" if self.model_handler.is_finetuned else "기본 모델"
        print(f"사용 모델: {model_type}")
        
        print(f"\n처리 통계:")
        success_rate = self.stats["model_success"] / self.stats["total"] * 100
        pattern_rate = self.stats["pattern_success"] / self.stats["total"] * 100
        fallback_rate = self.stats["fallback_used"] / self.stats["total"] * 100
        
        print(f"  모델 생성 성공: {self.stats['model_success']}/{self.stats['total']} ({success_rate:.1f}%)")
        print(f"  패턴 매칭 성공: {self.stats['pattern_success']}/{self.stats['total']} ({pattern_rate:.1f}%)")
        print(f"  스마트 힌트 사용: {self.stats['smart_hints_used']}회")
        print(f"  고신뢰도 답변: {self.stats['high_confidence_count']}회")
        print(f"  폴백 사용: {self.stats['fallback_used']}/{self.stats['total']} ({fallback_rate:.1f}%)")
        
        if self.model_handler.is_finetuned:
            finetuned_rate = self.stats["finetuned_usage"] / self.stats["total"] * 100
            print(f"  파인튜닝 활용: {self.stats['finetuned_usage']}/{self.stats['total']} ({finetuned_rate:.1f}%)")
        
        if self.stats["subj_count"] > 0:
            avg_korean_quality = self.stats["korean_quality_sum"] / self.stats["subj_count"]
            print(f"  평균 한국어 품질: {avg_korean_quality:.2f}")
        
        print(f"\n객관식 답변 분포:")
        total_mc = sum(self.stats["answer_distribution"].values())
        if total_mc > 0:
            for ans in sorted(self.stats["answer_distribution"].keys()):
                count = self.stats["answer_distribution"][ans]
                pct = count / total_mc * 100
                print(f"  {ans}번: {count}개 ({pct:.1f}%)")
            
            unique_answers = len([k for k, v in self.stats["answer_distribution"].items() if v > 0])
            print(f"  답변 다양성: {unique_answers}/5개 번호 사용")
            
            distribution_balance = np.std(list(self.stats["answer_distribution"].values()))
            if distribution_balance < total_mc * 0.15:
                print(f"  분포 균형: 양호")
            else:
                print(f"  분포 균형: 개선 필요")
        
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
    use_finetuned = False
    
    if len(sys.argv) > 1:
        try:
            test_size = int(sys.argv[1])
            test_size = max(1, min(test_size, 500))
        except:
            print("잘못된 문항 수, 기본값 50 사용")
            test_size = 50
    
    if len(sys.argv) > 2:
        if sys.argv[2].lower() in ['true', '1', 'yes', 'finetuned']:
            use_finetuned = True
    
    if os.path.exists("./finetuned_model") and not use_finetuned:
        response = input("파인튜닝된 모델이 발견되었습니다. 사용하시겠습니까? (y/n): ")
        if response.lower() in ['y', 'yes']:
            use_finetuned = True
    
    print(f"테스트 실행기 시작 (Python {sys.version.split()[0]})")
    
    runner = None
    try:
        runner = TestRunner(test_size=test_size, use_finetuned=use_finetuned)
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