# inference.py
"""
최종 추론 실행 파일 - 수정된 버전
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
warnings.filterwarnings("ignore")

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from model_handler import OptimizedModelHandler
from data_processor import IntelligentDataProcessor
from prompt_engineering import AdvancedPromptEngineer
from knowledge_base import FinancialSecurityKnowledgeBase

class HighPerformanceInferenceEngine:
    """추론 엔진 - 수정된 버전"""
    
    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.start_time = time.time()
        self.time_limit = 4.5 * 3600  # 4시간 30분
        
        print("모델 초기화 중...")
        
        # 컴포넌트 초기화
        self.model_handler = OptimizedModelHandler(**model_config)
        self.data_processor = IntelligentDataProcessor()
        self.prompt_engineer = AdvancedPromptEngineer()
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        
        # 성능 추적
        self.performance_stats = {
            "total_questions": 0,
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
            "processing_times": [],
            "strategy_usage": {},
            "retry_count": 0
        }
        
        print("초기화 완료")
    
    def execute_inference(self, test_file: str, submission_file: str, 
                         output_file: str = "./final_submission.csv") -> Dict:
        """메인 추론 실행"""
        
        # 데이터 로드
        test_df, sample_submission = self._load_data(test_file, submission_file)
        questions = test_df['Question'].tolist()
        question_ids = test_df['ID'].tolist()
        
        # 질문 분석 및 전략 수립
        strategies = self._analyze_and_strategize(questions)
        
        print(f"문항 분석 완료: 총 {len(questions)}개")
        print(f"선택형: {sum(1 for s in strategies if s['type'] == 'multiple_choice')}개")
        print(f"서술형: {sum(1 for s in strategies if s['type'] == 'subjective')}개")
        
        # 작은 배치 크기로 시작 (안정성 우선)
        batch_size = 2  # 문제 해결을 위해 크기 축소
        
        # 추론 실행
        predictions = self._execute_strategic_inference(
            questions, question_ids, strategies, batch_size
        )
        
        # 결과 저장 및 분석
        results = self._save_and_analyze_results(
            predictions, sample_submission, output_file
        )
        
        return results
    
    def _load_data(self, test_file: str, submission_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터 로드 및 검증"""
        try:
            test_df = pd.read_csv(test_file)
            sample_submission = pd.read_csv(submission_file)
            
            print(f"데이터 로드 완료: {len(test_df)}개 문항")
            
            # 데이터 무결성 검증
            assert len(test_df) == len(sample_submission), "데이터 크기 불일치"
            assert 'Question' in test_df.columns, "Question 컬럼 없음"
            assert 'ID' in test_df.columns, "ID 컬럼 없음"
            
            return test_df, sample_submission
            
        except Exception as e:
            print(f"데이터 로딩 오류: {e}")
            sys.exit(1)
    
    def _analyze_and_strategize(self, questions: List[str]) -> List[Dict]:
        """질문 분석 및 전략 수립"""
        strategies = []
        
        for i, question in enumerate(questions):
            # 기본 구조 분석
            structure = self.data_processor.analyze_question_structure(question)
            
            # 지식 베이스 분석
            knowledge_analysis = self.knowledge_base.analyze_question(question)
            
            # 전략 결정
            strategy = self._determine_strategy(structure, knowledge_analysis, i)
            
            strategies.append({
                "index": i,
                "type": structure["question_type"],
                "strategy": strategy,
                "difficulty": self._estimate_difficulty(structure, knowledge_analysis),
                "domain": knowledge_analysis.get("domain", ["general"]),
                "negative": structure.get("has_negative", False),
                "priority": self._calculate_priority(structure, knowledge_analysis)
            })
        
        # 우선순위 정렬 (어려운 문제 먼저 처리)
        strategies.sort(key=lambda x: (-x["priority"], x["index"]))
        
        return strategies
    
    def _determine_strategy(self, structure: Dict, knowledge_analysis: Dict, index: int) -> str:
        """최적 전략 결정"""
        
        # 도메인 전문성 기반
        domains = knowledge_analysis.get("domain", [])
        if "개인정보보호" in domains or "전자금융" in domains:
            if index < 50:  # 초기 문제들은 더 정교한 전략
                return "expert_few_shot"
            else:
                return "expert_analysis"
        
        # 법령 관련 문제
        if knowledge_analysis.get("relevant_laws"):
            return "law_focused"
        
        # 부정형 문제
        if structure.get("has_negative", False):
            return "negative_specialized"
        
        # 복잡한 기술 문제
        if structure["question_type"] == "subjective":
            return "chain_of_thought"
        
        # 기본 전략
        return "expert_analysis"
    
    def _estimate_difficulty(self, structure: Dict, knowledge_analysis: Dict) -> int:
        """난이도 추정 (1-5)"""
        difficulty = 1
        
        # 구조적 복잡성
        if structure["question_type"] == "subjective":
            difficulty += 1
        
        if structure.get("has_negative", False):
            difficulty += 1
        
        # 도메인 복잡성
        domains = knowledge_analysis.get("domain", [])
        if len(domains) > 1:
            difficulty += 1
        
        # 법령 관련
        if knowledge_analysis.get("relevant_laws"):
            difficulty += 1
        
        return min(difficulty, 5)
    
    def _calculate_priority(self, structure: Dict, knowledge_analysis: Dict) -> float:
        """처리 우선순위 계산"""
        priority = 1.0
        
        # 확신도가 높을 것으로 예상되는 문제 우선
        domains = knowledge_analysis.get("domain", [])
        if "개인정보보호" in domains:
            priority += 2.0
        if "전자금융" in domains:
            priority += 1.5
        
        # 명확한 법령 문제
        if knowledge_analysis.get("relevant_laws"):
            priority += 1.0
        
        # 정의 문제 (비교적 명확)
        if knowledge_analysis.get("question_type") == "정의_문제":
            priority += 0.5
        
        return priority
    
    def _execute_strategic_inference(self, questions: List[str], question_ids: List[str],
                                   strategies: List[Dict], batch_size: int) -> List[str]:
        """전략적 추론 실행 - 안전 모드"""
        
        predictions = [""] * len(questions)  # 원래 순서 유지
        processed_count = 0
        
        # 전략별 그룹화
        strategy_groups = self._group_by_strategy(strategies)
        
        print("추론 시작...")
        
        with tqdm(total=len(questions), desc="추론 진행") as pbar:
            
            for strategy_name, group_strategies in strategy_groups.items():
                print(f"\n전략 '{strategy_name}': {len(group_strategies)}개")
                
                # 개별 처리로 변경 (안정성 우선)
                for strategy_info in group_strategies:
                    
                    # 시간 제한 확인
                    if self._check_time_limit():
                        print("시간 제한 접근 - 빠른 처리 모드 전환")
                        remaining = self._process_remaining_quickly(
                            questions, predictions, strategies, processed_count
                        )
                        return remaining
                    
                    try:
                        # 개별 문제 처리
                        question = questions[strategy_info["index"]]
                        
                        # 타임아웃 설정으로 안전한 추론
                        result = self._process_single_question_safe(
                            question, strategy_info, strategy_name, timeout=30
                        )
                        
                        # 결과 저장 (원래 순서)
                        original_index = strategy_info["index"]
                        predictions[original_index] = result
                        processed_count += 1
                        pbar.update(1)
                        
                        # 10개마다 상태 출력
                        if processed_count % 10 == 0:
                            print(f"처리 완료: {processed_count}/{len(questions)}")
                        
                        # 메모리 정리
                        if processed_count % 50 == 0:
                            torch.cuda.empty_cache()
                            gc.collect()
                    
                    except Exception as e:
                        print(f"문제 {strategy_info['index']} 처리 오류: {e}")
                        # 폴백 답안
                        fallback_answer = self._create_fallback_answer(strategy_info["type"])
                        predictions[strategy_info["index"]] = fallback_answer
                        processed_count += 1
                        pbar.update(1)
        
        return predictions
    
    def _process_single_question_safe(self, question: str, strategy_info: Dict, 
                                    strategy_name: str, timeout: int = 30) -> str:
        """단일 문제 안전 처리"""
        
        try:
            # 프롬프트 생성
            prompt = self._create_strategic_prompt(question, strategy_info, strategy_name)
            optimized_prompt = self.prompt_engineer.optimize_for_model(
                prompt, self.model_config["model_name"]
            )
            
            # 안전한 생성 설정
            generation_kwargs = {
                "max_new_tokens": 256,  # 토큰 제한
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "pad_token_id": self.model_handler.tokenizer.eos_token_id,
                # early_stopping 제거
            }
            
            # 모델 추론 (타임아웃 적용)
            start_time = time.time()
            result = self.model_handler.generate_safe_response(
                optimized_prompt, 
                strategy_info["type"], 
                generation_kwargs,
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            if elapsed > 25:  # 25초 이상 걸리면 경고
                print(f"⚠️ 긴 처리 시간: {elapsed:.1f}초")
            
            # 후처리
            if hasattr(result, 'response'):
                raw_response = result.response
            else:
                raw_response = str(result)
            
            final_answer = self.data_processor.post_process_answer(
                raw_response, question, strategy_info["type"]
            )
            
            return final_answer
            
        except Exception as e:
            print(f"단일 처리 오류: {e}")
            return self._create_fallback_answer(strategy_info["type"])
    
    def _group_by_strategy(self, strategies: List[Dict]) -> Dict[str, List[Dict]]:
        """전략별 그룹화"""
        groups = {}
        for strategy_info in strategies:
            strategy_name = strategy_info["strategy"]
            if strategy_name not in groups:
                groups[strategy_name] = []
            groups[strategy_name].append(strategy_info)
        
        return groups
    
    def _create_strategic_prompt(self, question: str, strategy_info: Dict, 
                               strategy_name: str) -> str:
        """전략별 프롬프트 생성"""
        
        question_type = strategy_info["type"]
        
        if strategy_name == "expert_few_shot":
            return self.prompt_engineer.create_few_shot_expert_prompt(question, question_type)
        
        elif strategy_name == "law_focused":
            return self.prompt_engineer.create_expert_prompt(
                question, question_type, strategy="law_focused"
            )
        
        elif strategy_name == "chain_of_thought":
            return self.prompt_engineer.create_chain_of_thought_prompt(question, question_type)
        
        elif strategy_name == "negative_specialized":
            # 부정형 문제 특화 프롬프트
            base_prompt = self.prompt_engineer.create_expert_prompt(question, question_type)
            return f"{base_prompt}\n\n⚠️ 특별 주의: 이 문제는 '해당하지 않는' 또는 '틀린' 것을 찾는 부정형 문제입니다. 신중히 분석하세요."
        
        else:  # expert_analysis
            return self.prompt_engineer.create_expert_prompt(question, question_type)
    
    def _create_fallback_answer(self, question_type: str) -> str:
        """폴백 답안 생성"""
        if question_type == "multiple_choice":
            return "2"  # 가장 안전한 선택
        else:
            return "금융보안 정책에 따른 적절한 조치가 필요합니다."
    
    def _check_time_limit(self) -> bool:
        """시간 제한 확인"""
        elapsed = time.time() - self.start_time
        return elapsed > (self.time_limit * 0.9)  # 90% 지점에서 경고
    
    def _process_remaining_quickly(self, questions: List[str], predictions: List[str],
                                 strategies: List[Dict], processed_count: int) -> List[str]:
        """남은 문제 빠른 처리"""
        print("빠른 처리 모드")
        
        for i, prediction in enumerate(predictions):
            if prediction == "":  # 아직 처리되지 않은 문제
                question = questions[i]
                structure = self.data_processor.analyze_question_structure(question)
                
                if structure["question_type"] == "multiple_choice":
                    # 간단한 휴리스틱 사용
                    if structure.get("has_negative", False):
                        predictions[i] = "1"  # 부정형은 보통 1번이 명확히 틀림
                    else:
                        predictions[i] = "2"  # 일반적으로 2번이 안전
                else:
                    predictions[i] = "금융보안 정책에 따른 적절한 조치가 필요합니다."
        
        return predictions
    
    def _save_and_analyze_results(self, predictions: List[str], 
                                sample_submission: pd.DataFrame, 
                                output_file: str) -> Dict:
        """결과 저장 및 분석"""
        
        try:
            # 결과 저장
            sample_submission['Answer'] = predictions
            sample_submission.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 최종 통계
            total_time = time.time() - self.start_time
            
            # 답변 분포 분석
            mc_answers = [pred for pred in predictions 
                         if pred.strip() and pred.strip().isdigit()]
            
            distribution = {}
            for answer in mc_answers:
                choice = answer.strip()
                distribution[choice] = distribution.get(choice, 0) + 1
            
            results = {
                "output_file": output_file,
                "total_questions": len(predictions),
                "total_time_minutes": total_time / 60,
                "avg_time_per_question": total_time / len(predictions),
                "answer_distribution": distribution,
                "strategy_usage": self.performance_stats["strategy_usage"],
                "retry_count": self.performance_stats["retry_count"],
                "success": True
            }
            
            # 결과 출력
            print("\n=== 완료 ===")
            print(f"처리: {len(predictions)}개")
            print(f"시간: {total_time/60:.1f}분")
            print(f"평균: {total_time/len(predictions):.1f}초/문항")
            if self.performance_stats["retry_count"] > 0:
                print(f"재시도: {self.performance_stats['retry_count']}회")
            
            if mc_answers:
                print("\n선택형 분포:")
                total_mc = len(mc_answers)
                for choice, count in sorted(distribution.items()):
                    pct = (count / total_mc) * 100
                    print(f"  {choice}번: {count}개 ({pct:.1f}%)")
                
                # 편향 체크
                max_choice = max(distribution, key=distribution.get)
                max_pct = (distribution[max_choice] / total_mc) * 100
                if max_pct > 50:
                    print(f"⚠️  편향: {max_choice}번 {max_pct:.1f}%")
                else:
                    print("✅ 분포 양호")
            
            print(f"\n결과: {output_file}")
            print("완료")
            
            return results
            
        except Exception as e:
            print(f"결과 저장 오류: {e}")
            return {"success": False, "error": str(e)}
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self.model_handler.cleanup()
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"정리 중 오류: {e}")

def main():
    """메인 함수"""
    
    # GPU 확인
    if not torch.cuda.is_available():
        print("오류: CUDA 사용 불가")
        sys.exit(1)
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_memory:.1f}GB")
    
    # 파일 확인
    test_file = "./test.csv"
    submission_file = "./sample_submission.csv"
    
    if not os.path.exists(test_file) or not os.path.exists(submission_file):
        print("오류: 데이터 파일 없음")
        sys.exit(1)
    
    # 모델 설정 (안정성 우선)
    model_config = {
        "model_name": "upstage/SOLAR-10.7B-Instruct-v1.0",
        "device": "cuda",
        "load_in_4bit": False,  # RTX 4090에서는 16bit 사용
        "max_memory_gb": 20     # 여유 있게 설정
    }
    
    # 추론 엔진 초기화 및 실행
    engine = None
    try:
        engine = HighPerformanceInferenceEngine(model_config)
        results = engine.execute_inference(test_file, submission_file)
        
        if results["success"]:
            avg_time = results['avg_time_per_question']
            retry_count = results['retry_count']
            print(f"완료: 평균 {avg_time:.1f}초/문항, 재시도 {retry_count}회")
        
    except KeyboardInterrupt:
        print("추론 중단")
    except Exception as e:
        print(f"오류: {e}")
    finally:
        if engine:
            engine.cleanup()

if __name__ == "__main__":
    main()#