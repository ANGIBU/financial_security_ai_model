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
        print("\n시스템 초기화 중...")
        engine = FinancialAIInference(verbose=False)  # 초기화시 verbose=False로 설정
        
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
        
        # 답변 검증 (옵션)
        if test_size <= 10:
            validate_answers(output_file)
        
        return True
        
    except Exception as e:
        print(f"테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if engine:
            engine.cleanup()

def validate_answers(output_file: str):
    """답변 유효성 검증"""
    try:
        import pandas as pd
        result_df = pd.read_csv(output_file)
        
        # 객관식 답변 검증
        mc_answers = []
        subj_answers = []
        
        for idx, row in result_df.iterrows():
            answer = str(row['Answer']).strip()
            
            # 객관식 답변 (1-5 숫자)
            if answer in ['1', '2', '3', '4', '5']:
                mc_answers.append(answer)
            else:
                # 주관식 답변 검증
                korean_ratio = calculate_korean_ratio(answer)
                english_ratio = calculate_english_ratio(answer)
                
                if korean_ratio >= 0.8 and english_ratio <= 0.1:
                    subj_answers.append((row['ID'], "적합", f"한국어 {korean_ratio*100:.0f}%"))
                else:
                    subj_answers.append((row['ID'], "부적합", f"한국어 {korean_ratio*100:.0f}%, 영어 {english_ratio*100:.0f}%"))
        
        # 객관식 분포 출력
        if mc_answers:
            print(f"객관식 문항: {len(mc_answers)}개")
            for num in ['1', '2', '3', '4', '5']:
                count = mc_answers.count(num)
                percentage = (count / len(mc_answers)) * 100 if mc_answers else 0
                print(f"  {num}번: {count}개 ({percentage:.1f}%)")
        
        # 주관식 검증 결과
        if subj_answers:
            print(f"\n주관식 문항: {len(subj_answers)}개")
            suitable_count = sum(1 for _, status, _ in subj_answers if status == "적합")
            print(f"  한국어 적합: {suitable_count}개 ({suitable_count/len(subj_answers)*100:.1f}%)")
            
            # 부적합 답변 상세
            unsuitable = [item for item in subj_answers if item[1] == "부적합"]
            if unsuitable:
                print("  부적합 답변:")
                for id, status, detail in unsuitable[:5]:  # 최대 5개만 표시
                    print(f"    {id}: {detail}")
        
    except Exception as e:
        print(f"답변 검증 오류: {e}")

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

    print("테스트 완료")
    
    # 핵심 정보 출력
    total_time_minutes = results['total_time'] / 60
    
    print(f"\n기본 정보:")
    print(f"  처리 시간: {total_time_minutes:.1f}분")
    print(f"  처리 문항: {results['total_questions']}개")
    print(f"  결과 파일: {output_file}")

def select_test_size():
    """테스트 문항 수 선택"""
    print("\n테스트할 문항 수를 선택하세요:")
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

def main():
    """메인 함수"""
    # 테스트 크기 선택
    test_size = select_test_size()
    
    print(f"\n선택된 테스트: {test_size}문항")
    print("한국어 전용 답변 모드로 실행됩니다.")
    
    success = run_test(test_size, verbose=True)
    
    if success:
        print("\n테스트가 성공적으로 완료되었습니다.")
    else:
        print("\n테스트 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()