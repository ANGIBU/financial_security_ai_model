# test_runner.py

"""
테스트 실행기
- 시스템 테스트 실행
- 성능 측정 및 분석
- 결과 검증
- 한국어 준수율 검증
"""

import os
import sys
import time
import re
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
            temp_submission = submission_df.head(test_size).copy()
            
            # 통일된 출력 파일명
            output_file = "./test_result.csv"
                
            results = engine.execute_inference_with_data(
                test_df, 
                temp_submission, 
                output_file
            )
        else:
            output_file = "./test_result.csv"
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

def calculate_korean_ratio(text: str) -> float:
    """한국어 비율 계산"""
    if not text:
        return 0.0
    
    korean_chars = len(re.findall(r'[가-힣]', text))
    total_chars = len(re.sub(r'[^\w가-힣]', '', text))
    
    if total_chars == 0:
        return 0.0
    
    return korean_chars / total_chars

def calculate_english_ratio(text: str) -> float:
    """영어 비율 계산"""
    if not text:
        return 0.0
    
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total_chars = len(re.sub(r'[^\w가-힣]', '', text))
    
    if total_chars == 0:
        return 0.0
    
    return english_chars / total_chars

def print_test_results(results: dict, output_file: str, test_size: int):
    """테스트 결과 출력"""
    
    print("\n" + "="*50)
    print("테스트 완료")
    print("="*50)
    
    # 핵심 통계만 출력
    total_time_minutes = results['total_time'] / 60
    mc_reliability = (results['model_success_rate'] if results['mc_count'] > 0 else 0)
    subj_reliability = (results['avg_quality_score'] * 100 if results['subj_count'] > 0 else 0)
    
    print(f"총 처리 시간: {total_time_minutes:.1f}분")
    print(f"처리 문항: {results['total_questions']}개")
    print(f"객관식 평균 신뢰도: {mc_reliability:.1f}%")
    print(f"주관식 평균 신뢰도: {subj_reliability:.1f}%")
    print(f"한국어 준수율: {results['korean_compliance_rate']:.1f}%")
    
    # 객관식 답변 분포 (간단히)
    if results['mc_count'] > 0:
        distribution = results['answer_distribution']
        used_numbers = len([v for v in distribution.values() if v > 0])
        print(f"객관식 답변 다양성: {used_numbers}/5개 번호 사용")
    
    print(f"결과 파일: {output_file}")
    print("="*50)

def select_test_size():
    """테스트 문항 수 선택"""
    print("테스트할 문항 수를 선택하세요:")
    print("1. 5문항 (빠른 테스트)")
    print("2. 10문항 (기본 테스트)")
    print("3. 50문항 (전체 테스트)")
    print()
    
    while True:
        try:
            choice = input("선택 (1-3): ").strip()
            
            if choice == "1":
                return 5
            elif choice == "2":
                return 10
            elif choice == "3":
                return 50
            else:
                print("잘못된 선택입니다. 1, 2, 3 중 하나를 입력하세요.")
                
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            sys.exit(0)
        except Exception:
            print("잘못된 입력입니다. 다시 시도하세요.")

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
    print("금융보안 AI 테스트 시스템")
    print(f"Python 버전: {sys.version.split()[0]}")
    print()
    
    # 테스트 크기 선택
    test_size = select_test_size()
    
    print(f"선택된 테스트: {test_size}문항")
    print("한국어 전용 답변 모드로 실행됩니다.")
    print()
    
    success = run_test(test_size, verbose=False)
    
    if not success:
        print("테스트 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()