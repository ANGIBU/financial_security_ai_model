# test_runner.py

"""
테스트 실행기
- 시스템 테스트 실행
- 성능 측정 및 분석
- 결과 검증
"""

import os
import sys
import time
from pathlib import Path

# 현재 디렉토리 설정
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from inference import FinancialAIInference

def run_test(test_size: int = 50, verbose: bool = True):
    """테스트 실행"""
    
    print(f"테스트 실행 ({test_size}문항)")
    
    # 파일 존재 확인
    test_file = "./test.csv"
    submission_file = "./sample_submission.csv"
    
    for file_path in [test_file, submission_file]:
        if not os.path.exists(file_path):
            print(f"오류: {file_path} 파일이 없습니다")
            return False
    
    engine = None
    try:
        # AI 엔진 초기화
        engine = FinancialAIInference(verbose=verbose)
        
        # 테스트 데이터 준비
        import pandas as pd
        test_df = pd.read_csv(test_file)
        submission_df = pd.read_csv(submission_file)
        
        # 지정된 크기로 제한
        if len(test_df) > test_size:
            test_df = test_df.head(test_size)
            
            # 테스트 파일만 생성
            test_df.to_csv("./test_temp.csv", index=False)
            
            # submission 파일은 메모리에서 처리
            temp_submission = submission_df.head(test_size).copy()
            
            output_file = f"./test_result_{test_size}.csv"
            results = engine.execute_inference_with_data(
                test_df, 
                temp_submission, 
                output_file
            )
            
            # 임시 파일 정리
            if os.path.exists("./test_temp.csv"):
                os.remove("./test_temp.csv")
        else:
            output_file = f"./test_result_{len(test_df)}.csv"
            results = engine.execute_inference(
                test_file,
                submission_file,
                output_file
            )
        
        # 결과 분석
        print_test_results(results, output_file, test_size)
        
        return True
        
    except Exception as e:
        print(f"테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if engine:
            engine.cleanup()

def print_test_results(results: dict, output_file: str, test_size: int):
    """테스트 결과 출력"""
    
    print("\n" + "=" * 60)
    print("테스트 결과 분석")
    print("=" * 60)
    
    # 기본 통계
    print(f"처리 완료: {results['total_questions']}문항")
    print(f"객관식: {results['mc_count']}개, 주관식: {results['subj_count']}개")
    print(f"모델 성공률: {results['model_success_rate']:.1f}%")
    print(f"평균 처리시간: {results['avg_processing_time']:.2f}초/문항")
    print(f"총 소요시간: {results['total_time']:.1f}초")
    
    # 도메인별 분포
    if results['domain_stats']:
        print(f"\n도메인별 처리 현황:")
        for domain, count in results['domain_stats'].items():
            pct = (count / results['total_questions']) * 100
            print(f"  {domain}: {count}개 ({pct:.1f}%)")
    
    # 난이도별 분포
    if results['difficulty_stats']:
        print(f"\n난이도별 처리 현황:")
        for difficulty, count in results['difficulty_stats'].items():
            pct = (count / results['total_questions']) * 100
            print(f"  {difficulty}: {count}개 ({pct:.1f}%)")
    
    # 객관식 답변 분포 분석
    distribution = results['answer_distribution']
    total_mc = sum(distribution.values())
    
    if total_mc > 0:
        print(f"\n객관식 답변 분포:")
        for num in range(1, 6):
            count = distribution[str(num)]
            pct = (count / total_mc) * 100
            print(f"  {num}번: {count}개 ({pct:.1f}%)")
        
        # 다양성 평가
        used_numbers = len([v for v in distribution.values() if v > 0])
        diversity_score = calculate_diversity_score(distribution, total_mc)
        
        if used_numbers >= 4 and diversity_score > 0.8:
            diversity_status = "우수"
        elif used_numbers >= 3 and diversity_score > 0.6:
            diversity_status = "양호"
        else:
            diversity_status = "개선필요"
        
        print(f"  답변 다양성: {diversity_status} ({used_numbers}/5개 번호 사용)")
        print(f"  다양성 점수: {diversity_score:.2f}")
    
    # 성능 평가
    print(f"\n성능 평가:")
    
    # 모델 성능
    if results['model_success_rate'] >= 80:
        model_grade = "A"
    elif results['model_success_rate'] >= 60:
        model_grade = "B"
    elif results['model_success_rate'] >= 40:
        model_grade = "C"
    else:
        model_grade = "D"
    
    print(f"모델 성능: {model_grade}등급 ({results['model_success_rate']:.1f}%)")
    
    # 처리 속도
    if results['avg_processing_time'] <= 10:
        speed_grade = "A"
    elif results['avg_processing_time'] <= 20:
        speed_grade = "B"
    elif results['avg_processing_time'] <= 30:
        speed_grade = "C"
    else:
        speed_grade = "D"
    
    print(f"처리 속도: {speed_grade}등급 ({results['avg_processing_time']:.2f}초/문항)")
    
    # 시간 효율성
    expected_time = test_size * 30  # 30초/문항 기준
    efficiency = (expected_time / results['total_time']) * 100
    print(f"시간 효율성: {efficiency:.1f}%")
    
    print(f"\n결과 파일: {output_file}")
    
    # 파일 내용 검증
    validate_output_file(output_file, results)

def calculate_diversity_score(distribution: dict, total: int) -> float:
    """다양성 점수 계산"""
    if total == 0:
        return 0.0
    
    # 이상적인 분포: 각 번호가 20%씩
    ideal_ratio = 0.2
    actual_ratios = [distribution[str(i)] / total for i in range(1, 6)]
    
    # 편차 계산
    deviations = [abs(ratio - ideal_ratio) for ratio in actual_ratios]
    avg_deviation = sum(deviations) / len(deviations)
    
    # 점수 계산 (편차가 작을수록 높은 점수)
    diversity_score = max(0, 1 - (avg_deviation / ideal_ratio))
    
    return diversity_score

def validate_output_file(output_file: str, results: dict):
    """출력 파일 검증"""
    try:
        import pandas as pd
        result_df = pd.read_csv(output_file)
        
        print(f"\n파일 검증:")
        print(f"  총 답변 수: {len(result_df)}개")
        
        # 답변 형식 검증
        mc_answers = 0
        subj_answers = 0
        invalid_answers = 0
        
        for answer in result_df['Answer']:
            answer_str = str(answer).strip()
            if answer_str in ['1', '2', '3', '4', '5']:
                mc_answers += 1
            elif len(answer_str) > 10:
                subj_answers += 1
            else:
                invalid_answers += 1
        
        print(f"  객관식 답변: {mc_answers}개")
        print(f"  주관식 답변: {subj_answers}개")
        
        if invalid_answers > 0:
            print(f"  유효하지 않은 답변: {invalid_answers}개")
        
        # 예상 비율과 비교
        expected_mc = results['mc_count']
        expected_subj = results['subj_count']
        
        mc_accuracy = (mc_answers / expected_mc * 100) if expected_mc > 0 else 0
        subj_accuracy = (subj_answers / expected_subj * 100) if expected_subj > 0 else 0
        
        print(f"  객관식 정확도: {mc_accuracy:.1f}%")
        print(f"  주관식 정확도: {subj_accuracy:.1f}%")
        
        # 품질 평가
        if mc_answers > 0:
            # 객관식 분포 확인
            mc_dist = {}
            for answer in result_df['Answer']:
                if str(answer).strip() in ['1', '2', '3', '4', '5']:
                    key = str(answer).strip()
                    mc_dist[key] = mc_dist.get(key, 0) + 1
            
            # 편향 확인
            if len(mc_dist) == 1:
                print("  경고: 모든 객관식 답변이 동일한 번호")
            elif len(mc_dist) >= 4:
                print("  객관식 답변 분포 양호")
        
        if subj_answers > 0:
            # 주관식 답변 길이 확인
            subj_lengths = []
            for answer in result_df['Answer']:
                answer_str = str(answer).strip()
                if len(answer_str) > 10:
                    subj_lengths.append(len(answer_str))
            
            if subj_lengths:
                avg_length = sum(subj_lengths) / len(subj_lengths)
                print(f"  주관식 평균 길이: {avg_length:.0f}자")
                
                if avg_length < 20:
                    print("  경고: 주관식 답변이 너무 짧음")
                elif avg_length > 500:
                    print("  경고: 주관식 답변이 너무 김")
                else:
                    print("  주관식 답변 길이 적절")
        
    except Exception as e:
        print(f"파일 검증 오류: {e}")

def print_progress_bar(current: int, total: int, start_time: float, bar_length: int = 50):
    """진행률 게이지바 출력"""
    progress = current / total
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    # 시간 계산
    elapsed = time.time() - start_time
    if current > 0:
        avg_time_per_item = elapsed / current
        remaining_items = total - current
        eta = avg_time_per_item * remaining_items
        eta_minutes = int(eta // 60)
        eta_seconds = int(eta % 60)
        eta_str = f"{eta_minutes:02d}:{eta_seconds:02d}"
    else:
        eta_str = "--:--"
    
    # 진행률 출력
    percent = progress * 100
    print(f"\r진행: [{bar}] {current}/{total} ({percent:.1f}%) - 남은시간: {eta_str}", end='', flush=True)

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='금융보안 AI 테스트')
    parser.add_argument('--size', type=int, default=50, help='테스트할 문항 수 (기본: 50)')
    parser.add_argument('--verbose', action='store_true', help='상세 출력')
    
    args = parser.parse_args()
    
    # 테스트 크기 제한
    test_size = max(1, min(args.size, 515))
    
    print(f"Python 버전: {sys.version.split()[0]}")
    print(f"테스트 크기: {test_size}문항")
    
    success = run_test(test_size, args.verbose)
    
    if success:
        print("\n테스트 완료!")
    else:
        print("\n테스트 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()