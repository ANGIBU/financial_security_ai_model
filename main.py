# main.py
"""
개발/테스트용 메인 파일
"""

import os
import pandas as pd
import torch
import time
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from model_handler import OptimizedModelHandler
from data_processor import IntelligentDataProcessor
from prompt_engineering import AdvancedPromptEngineer
from knowledge_base import FinancialSecurityKnowledgeBase

class DevelopmentTester:
    """개발 및 테스트 클래스"""
    
    def __init__(self, model_config: dict):
        print("개발 시스템 초기화...")
        
        # 컴포넌트 초기화
        self.model_handler = OptimizedModelHandler(**model_config)
        self.data_processor = IntelligentDataProcessor()
        self.prompt_engineer = AdvancedPromptEngineer()
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        
        print("초기화 완료")
    
    def run_sample_test(self, sample_size: int = 10, strategy: str = "expert_analysis"):
        """샘플 테스트 실행"""
        
        # 데이터 로드
        test_df = pd.read_csv('./test.csv')
        sample_submission = pd.read_csv('./sample_submission.csv')
        
        print(f"\n샘플 테스트: {sample_size}개 문항")
        print(f"전략: {strategy}")
        
        # 샘플 선택 (다양한 유형 포함)
        sample_indices = self._select_diverse_samples(test_df, sample_size)
        
        results = []
        total_time = 0
        
        for idx in tqdm(sample_indices, desc="테스트 진행"):
            start_time = time.time()
            
            question = test_df.iloc[idx]['Question']
            question_id = test_df.iloc[idx]['ID']
            
            # 문제 분석
            structure = self.data_processor.analyze_question_structure(question)
            knowledge_analysis = self.knowledge_base.analyze_question(question)
            
            # 프롬프트 생성
            prompt = self._create_test_prompt(question, structure, strategy)
            optimized_prompt = self.prompt_engineer.optimize_for_model(
                prompt, self.model_handler.model_name
            )
            
            # 추론 실행
            try:
                inference_result = self.model_handler.generate_expert_response(
                    optimized_prompt, structure["question_type"]
                )
                
                # 후처리
                final_answer = self.data_processor.post_process_answer(
                    inference_result.response, question, structure["question_type"]
                )
                
                # 결과 저장
                elapsed = time.time() - start_time
                total_time += elapsed
                
                results.append({
                    "id": question_id,
                    "question": question[:100] + "..." if len(question) > 100 else question,
                    "type": structure["question_type"],
                    "domain": knowledge_analysis.get("domain", ["general"]),
                    "answer": final_answer,
                    "confidence": inference_result.confidence,
                    "reasoning_quality": inference_result.reasoning_quality,
                    "time": elapsed,
                    "negative": structure.get("has_negative", False)
                })
                
            except Exception as e:
                print(f"오류 발생 (ID: {question_id}): {e}")
                results.append({
                    "id": question_id,
                    "question": question[:100] + "...",
                    "type": structure["question_type"],
                    "domain": ["error"],
                    "answer": "오류",
                    "confidence": 0.0,
                    "reasoning_quality": 0.0,
                    "time": 0.0,
                    "negative": False
                })
        
        # 결과 분석
        self._analyze_test_results(results, total_time)
        
        return results
    
    def _select_diverse_samples(self, test_df: pd.DataFrame, sample_size: int) -> list:
        """다양한 유형의 샘플 선택"""
        indices = []
        
        # 다양한 패턴의 문제 선택
        patterns = [
            ("개인정보", 2),
            ("전자금융", 2), 
            ("보안", 2),
            ("암호화", 1),
            ("해당하지 않는", 1),  # 부정형
            ("적절하지 않은", 1),   # 부정형
        ]
        
        used_indices = set()
        
        for pattern, count in patterns:
            found = 0
            for i, question in enumerate(test_df['Question']):
                if pattern in question and i not in used_indices:
                    indices.append(i)
                    used_indices.add(i)
                    found += 1
                    if found >= count:
                        break
        
        # 부족한 만큼 랜덤 선택
        while len(indices) < sample_size:
            idx = len(indices) * 50  # 분산 선택
            if idx < len(test_df) and idx not in used_indices:
                indices.append(idx)
                used_indices.add(idx)
        
        return indices[:sample_size]
    
    def _create_test_prompt(self, question: str, structure: dict, strategy: str) -> str:
        """테스트용 프롬프트 생성"""
        
        question_type = structure["question_type"]
        
        if strategy == "expert_few_shot":
            return self.prompt_engineer.create_few_shot_expert_prompt(question, question_type)
        elif strategy == "chain_of_thought":
            return self.prompt_engineer.create_chain_of_thought_prompt(question, question_type)
        elif strategy == "law_focused":
            return self.prompt_engineer.create_expert_prompt(question, question_type, "law_focused")
        else:  # expert_analysis
            return self.prompt_engineer.create_expert_prompt(question, question_type)
    
    def _analyze_test_results(self, results: list, total_time: float):
        """테스트 결과 분석"""
        
        print(f"\n=== 결과 분석 ===")
        print(f"처리 시간: {total_time:.1f}초")
        print(f"평균 시간: {total_time/len(results):.1f}초/문항")
        
        # 유형별 분석
        mc_results = [r for r in results if r["type"] == "multiple_choice"]
        subj_results = [r for r in results if r["type"] == "subjective"]
        
        print(f"\n객관식: {len(mc_results)}개")
        print(f"주관식: {len(subj_results)}개")
        
        # 신뢰도 분석
        high_conf = len([r for r in results if r["confidence"] >= 0.7])
        medium_conf = len([r for r in results if 0.3 <= r["confidence"] < 0.7])
        low_conf = len([r for r in results if r["confidence"] < 0.3])
        
        print(f"\n신뢰도 분포:")
        print(f"  높음 (≥0.7): {high_conf}개 ({high_conf/len(results)*100:.1f}%)")
        print(f"  보통 (0.3-0.7): {medium_conf}개 ({medium_conf/len(results)*100:.1f}%)")
        print(f"  낮음 (<0.3): {low_conf}개 ({low_conf/len(results)*100:.1f}%)")
        
        # 객관식 답변 분포
        if mc_results:
            mc_answers = [r["answer"] for r in mc_results if r["answer"].isdigit()]
            if mc_answers:
                print(f"\n객관식 답변 분포:")
                for choice in "12345":
                    count = mc_answers.count(choice)
                    pct = (count / len(mc_answers)) * 100 if mc_answers else 0
                    print(f"  {choice}번: {count}개 ({pct:.1f}%)")
                
                # 편향 체크
                max_choice = max("12345", key=lambda x: mc_answers.count(x))
                max_pct = (mc_answers.count(max_choice) / len(mc_answers)) * 100
                if max_pct > 60:
                    print(f"⚠️  편향 감지: {max_choice}번 {max_pct:.1f}%")
                else:
                    print("✅ 답변 분포 양호")
        
        # 상세 결과 출력 (처음 5개)
        print(f"\n=== 상세 결과 (처음 5개) ===")
        for i, result in enumerate(results[:5], 1):
            print(f"\n{i}. {result['id']}")
            print(f"   문제: {result['question']}")
            print(f"   유형: {result['type']}")
            print(f"   도메인: {', '.join(result['domain'])}")
            print(f"   답변: {result['answer']}")
            print(f"   신뢰도: {result['confidence']:.2f}")
            print(f"   추론품질: {result['reasoning_quality']:.2f}")
            print(f"   처리시간: {result['time']:.1f}초")
        
        # 성능 예측
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        avg_reasoning = sum(r["reasoning_quality"] for r in results) / len(results)
        
        print(f"\n=== 성능 예측 ===")
        print(f"평균 신뢰도: {avg_confidence:.2f}")
        print(f"평균 추론품질: {avg_reasoning:.2f}")
        
        if avg_confidence >= 0.6 and avg_reasoning >= 0.5:
            print("예상 성능: 우수")
        elif avg_confidence >= 0.4 and avg_reasoning >= 0.3:
            print("예상 성능: 양호")
        else:
            print("예상 성능: 개선 필요")
    
    def run_strategy_comparison(self, sample_size: int = 20):
        """전략별 성능 비교"""
        
        strategies = ["expert_analysis", "expert_few_shot", "chain_of_thought", "law_focused"]
        
        print(f"\n=== 전략 비교 테스트 ({sample_size}개) ===")
        
        # 동일한 문제로 각 전략 테스트
        test_df = pd.read_csv('./test.csv')
        sample_indices = self._select_diverse_samples(test_df, sample_size)
        
        strategy_results = {}
        
        for strategy in strategies:
            print(f"\n전략 '{strategy}' 테스트 중...")
            
            strategy_scores = []
            strategy_times = []
            
            for idx in tqdm(sample_indices, desc=f"{strategy}"):
                question = test_df.iloc[idx]['Question']
                structure = self.data_processor.analyze_question_structure(question)
                
                start_time = time.time()
                try:
                    prompt = self._create_test_prompt(question, structure, strategy)
                    optimized_prompt = self.prompt_engineer.optimize_for_model(
                        prompt, self.model_handler.model_name
                    )
                    
                    result = self.model_handler.generate_expert_response(
                        optimized_prompt, structure["question_type"], max_attempts=1
                    )
                    
                    elapsed = time.time() - start_time
                    
                    strategy_scores.append(result.confidence * result.reasoning_quality)
                    strategy_times.append(elapsed)
                    
                except Exception:
                    strategy_scores.append(0.0)
                    strategy_times.append(time.time() - start_time)
            
            avg_score = sum(strategy_scores) / len(strategy_scores)
            avg_time = sum(strategy_times) / len(strategy_times)
            
            strategy_results[strategy] = {
                "avg_score": avg_score,
                "avg_time": avg_time,
                "success_rate": len([s for s in strategy_scores if s > 0.3]) / len(strategy_scores)
            }
        
        # 결과 출력
        print(f"\n=== 전략별 성능 ===")
        for strategy, metrics in strategy_results.items():
            print(f"\n{strategy}:")
            print(f"  점수: {metrics['avg_score']:.3f}")
            print(f"  시간: {metrics['avg_time']:.1f}초")
            print(f"  성공률: {metrics['success_rate']*100:.1f}%")
        
        # 최고 전략 추천
        best_strategy = max(strategy_results.keys(), 
                          key=lambda x: strategy_results[x]['avg_score'])
        
        print(f"\n최고 성능: {best_strategy}")
        print(f"점수: {strategy_results[best_strategy]['avg_score']:.3f}")
    
    def cleanup(self):
        """리소스 정리"""
        self.model_handler.cleanup()

def main():
    """개발용 메인 함수"""
    
    parser = argparse.ArgumentParser(description='개발 도구')
    parser.add_argument('--sample-size', type=int, default=10, 
                       help='테스트할 샘플 수 (기본: 10)')
    parser.add_argument('--strategy', type=str, default='expert_analysis',
                       choices=['expert_analysis', 'expert_few_shot', 'chain_of_thought', 'law_focused'],
                       help='테스트할 전략 (기본: expert_analysis)')
    parser.add_argument('--compare-strategies', action='store_true',
                       help='전략별 성능 비교 실행')
    parser.add_argument('--full-test', action='store_true',
                       help='전체 데이터로 테스트 (시간 오래 걸림)')
    
    args = parser.parse_args()
    
    # GPU 확인
    if not torch.cuda.is_available():
        print("CUDA 사용 불가")
        return
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_memory:.1f}GB")
    
    # 데이터 파일 확인
    if not os.path.exists('./test.csv') or not os.path.exists('./sample_submission.csv'):
        print("오류: 데이터 파일 없음")
        return
    
    # 모델 설정 (RTX 4090 24GB 최적화)
    max_memory = 22  # RTX 4090 24GB
    
    model_config = {
        "model_name": "upstage/SOLAR-10.7B-Instruct-v1.0",
        "device": "cuda",
        "load_in_4bit": False,  # RTX 4090에서는 16bit 사용
        "max_memory_gb": max_memory
    }
    
    # 테스터 초기화
    tester = None
    try:
        tester = DevelopmentTester(model_config)
        
        if args.compare_strategies:
            # 전략 비교
            tester.run_strategy_comparison(args.sample_size)
        elif args.full_test:
            # 전체 테스트
            print("전체 테스트: python inference.py 사용")
        else:
            # 기본 샘플 테스트
            tester.run_sample_test(args.sample_size, args.strategy)
        
        print("테스트 완료")
        
    except KeyboardInterrupt:
        print("테스트 중단")
    except Exception as e:
        print(f"오류: {e}")
    finally:
        if tester:
            tester.cleanup()

if __name__ == "__main__":
    main()