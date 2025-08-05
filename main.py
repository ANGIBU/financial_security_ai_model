# main.py
"""
개발/테스트용 메인 파일 - 최적화 버전
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
    """개발 및 테스트 클래스 - 최적화 버전"""
    
    def __init__(self, model_config: dict):
        print("개발 시스템 초기화...")
        
        # 컴포넌트 초기화
        self.model_handler = OptimizedModelHandler(**model_config)
        self.data_processor = IntelligentDataProcessor()
        self.prompt_engineer = AdvancedPromptEngineer()
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        
        print("초기화 완료")
    
    def run_accuracy_test(self, sample_size: int = 20):
        """정확도 중심 테스트"""
        
        # 데이터 로드
        test_df = pd.read_csv('./test.csv')
        
        print(f"\n정확도 테스트: {sample_size}개 문항")
        
        # 다양한 유형 샘플 선택
        sample_indices = self._select_diverse_samples(test_df, sample_size)
        
        results = []
        correct_patterns = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        
        for idx in tqdm(sample_indices, desc="테스트 진행"):
            question = test_df.iloc[idx]['Question']
            question_id = test_df.iloc[idx]['ID']
            
            # 문제 분석
            structure = self.data_processor.analyze_question_structure(question)
            analysis = self.knowledge_base.analyze_question(question)
            
            # 다양한 전략 테스트
            strategies = ["simple", "balanced", "complex"]
            best_answer = None
            best_confidence = 0
            
            for strategy in strategies:
                try:
                    prompt = self.prompt_engineer.create_expert_prompt(
                        question, structure["question_type"], strategy
                    )
                    optimized_prompt = self.prompt_engineer.optimize_for_model(
                        prompt, self.model_handler.model_name
                    )
                    
                    result = self.model_handler.generate_expert_response(
                        optimized_prompt, structure["question_type"], max_attempts=1
                    )
                    
                    processed_answer = self.data_processor.post_process_answer(
                        result.response, question, structure["question_type"]
                    )
                    
                    if result.confidence > best_confidence:
                        best_confidence = result.confidence
                        best_answer = processed_answer
                        
                except Exception as e:
                    continue
            
            if best_answer and structure["question_type"] == "multiple_choice":
                if best_answer in correct_patterns:
                    correct_patterns[best_answer] += 1
            
            results.append({
                "id": question_id,
                "question": question[:100] + "...",
                "answer": best_answer,
                "confidence": best_confidence,
                "type": structure["question_type"],
                "domain": analysis.get("domain", ["general"])
            })
        
        # 결과 분석
        self._analyze_accuracy_results(results, correct_patterns)
        
        return results
    
    def run_speed_test(self, sample_size: int = 50):
        """속도 최적화 테스트"""
        
        test_df = pd.read_csv('./test.csv')
        
        print(f"\n속도 테스트: {sample_size}개 문항")
        
        # 빠른 처리를 위한 간단한 문제 위주 선택
        sample_indices = list(range(min(sample_size, len(test_df))))
        
        start_time = time.time()
        results = []
        
        # 배치 처리 테스트
        batch_size = 10
        for i in tqdm(range(0, len(sample_indices), batch_size), desc="배치 처리"):
            batch_indices = sample_indices[i:i+batch_size]
            batch_questions = [test_df.iloc[idx]['Question'] for idx in batch_indices]
            
            # 빠른 프롬프트 생성
            prompts = []
            types = []
            for question in batch_questions:
                structure = self.data_processor.analyze_question_structure(question)
                prompt = self.prompt_engineer.create_simple_mc_prompt(question)
                optimized = self.prompt_engineer.optimize_for_model(
                    prompt, self.model_handler.model_name
                )
                prompts.append(optimized)
                types.append(structure["question_type"])
            
            # 배치 추론
            try:
                batch_results = self.model_handler.generate_batch_responses(
                    prompts, types, batch_size=len(prompts)
                )
                
                for j, result in enumerate(batch_results):
                    answer = self.data_processor.extract_mc_answer_fast(result.response)
                    results.append({
                        "id": test_df.iloc[batch_indices[j]]['ID'],
                        "answer": answer,
                        "time": result.inference_time
                    })
            except Exception as e:
                print(f"배치 처리 오류: {e}")
        
        total_time = time.time() - start_time
        
        print(f"\n총 처리 시간: {total_time:.1f}초")
        print(f"평균: {total_time/len(results):.2f}초/문항")
        print(f"예상 전체 시간: {(total_time/len(results)*515)/60:.1f}분")
        
        return results
    
    def _select_diverse_samples(self, test_df: pd.DataFrame, sample_size: int) -> list:
        """다양한 유형의 샘플 선택"""
        indices = []
        
        # 패턴별 선택
        patterns = [
            ("개인정보", min(5, sample_size//4)),
            ("전자금융", min(5, sample_size//4)),
            ("해당하지 않는", min(3, sample_size//5)),
            ("정의", min(3, sample_size//5)),
            ("법", min(4, sample_size//5))
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
        
        # 나머지는 랜덤
        remaining = sample_size - len(indices)
        for i in range(0, len(test_df), len(test_df) // remaining if remaining > 0 else 1):
            if i not in used_indices and len(indices) < sample_size:
                indices.append(i)
                used_indices.add(i)
        
        return indices[:sample_size]
    
    def _analyze_accuracy_results(self, results: list, patterns: dict):
        """정확도 결과 분석"""
        
        print(f"\n=== 정확도 분석 ===")
        
        # 타입별 분석
        mc_results = [r for r in results if r["type"] == "multiple_choice"]
        subj_results = [r for r in results if r["type"] == "subjective"]
        
        print(f"객관식: {len(mc_results)}개")
        print(f"주관식: {len(subj_results)}개")
        
        # 신뢰도 분포
        high_conf = len([r for r in results if r["confidence"] >= 0.7])
        print(f"\n높은 신뢰도 (≥0.7): {high_conf}개 ({high_conf/len(results)*100:.1f}%)")
        
        # 답변 분포
        if patterns:
            print(f"\n객관식 답변 분포:")
            total = sum(patterns.values())
            for choice, count in sorted(patterns.items()):
                pct = (count / total * 100) if total > 0 else 0
                print(f"  {choice}번: {count}개 ({pct:.1f}%)")
            
            # 편향 체크
            if total > 0:
                max_choice = max(patterns, key=patterns.get)
                max_pct = (patterns[max_choice] / total) * 100
                if max_pct > 50:
                    print(f"⚠️  편향 감지: {max_choice}번 {max_pct:.1f}%")
                else:
                    print("✅ 답변 분포 균형")
        
        # 도메인별 분석
        domain_stats = {}
        for result in results:
            for domain in result["domain"]:
                if domain not in domain_stats:
                    domain_stats[domain] = 0
                domain_stats[domain] += 1
        
        print(f"\n도메인별 분포:")
        for domain, count in sorted(domain_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {domain}: {count}개")
    
    def cleanup(self):
        """리소스 정리"""
        self.model_handler.cleanup()

def main():
    """개발용 메인 함수"""
    
    parser = argparse.ArgumentParser(description='개발 도구')
    parser.add_argument('--test-type', type=str, default='accuracy',
                       choices=['accuracy', 'speed', 'both'],
                       help='테스트 유형 (기본: accuracy)')
    parser.add_argument('--sample-size', type=int, default=20,
                       help='테스트할 샘플 수 (기본: 20)')
    
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
    model_config = {
        "model_name": "upstage/SOLAR-10.7B-Instruct-v1.0",
        "device": "cuda",
        "load_in_4bit": False,
        "max_memory_gb": 22
    }
    
    # 테스터 초기화
    tester = None
    try:
        tester = DevelopmentTester(model_config)
        
        if args.test_type == 'accuracy' or args.test_type == 'both':
            tester.run_accuracy_test(args.sample_size)
        
        if args.test_type == 'speed' or args.test_type == 'both':
            tester.run_speed_test(args.sample_size * 2)  # 속도 테스트는 더 많이
        
        print("\n테스트 완료")
        
    except KeyboardInterrupt:
        print("\n테스트 중단")
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if tester:
            tester.cleanup()

if __name__ == "__main__":
    main()