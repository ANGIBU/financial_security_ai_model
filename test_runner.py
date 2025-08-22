# test_runner.py

"""
테스트 실행기
- 시스템 테스트 실행
- 핵심 성능 지표 출력
"""

import os
import sys
from pathlib import Path

# 현재 디렉토리 설정
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# 설정 파일 import
from config import FILE_VALIDATION, DEFAULT_FILES
from inference import FinancialAIInference


def run_test(test_size: int = None, verbose: bool = True):
    """테스트 실행"""

    # 기본 테스트 크기 설정
    if test_size is None:
        test_size = 50

    # 파일 존재 확인
    test_file = DEFAULT_FILES["test_file"]
    submission_file = DEFAULT_FILES["submission_file"]

    for file_path in [test_file, submission_file]:
        if not os.path.exists(file_path):
            print(f"오류: {file_path} 파일이 없습니다")
            return False

    engine = None
    try:
        # AI 엔진 초기화
        print("\n시스템 초기화 중...")
        engine = FinancialAIInference(verbose=False)

        # 테스트 데이터 준비
        import pandas as pd

        test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["encoding"])
        submission_df = pd.read_csv(
            submission_file, encoding=FILE_VALIDATION["encoding"]
        )

        print(f"전체 데이터: {len(test_df)}개 문항")
        print(f"테스트 크기: {test_size}개 문항")

        # 지정된 크기로 제한
        if len(test_df) > test_size:
            test_df = test_df.head(test_size)
            temp_submission = submission_df.head(test_size).copy()

            output_file = DEFAULT_FILES["test_output_file"]
            results = engine.execute_inference_with_data(
                test_df, temp_submission, output_file
            )
        else:
            output_file = DEFAULT_FILES["test_output_file"]
            results = engine.execute_inference(test_file, submission_file, output_file)

        # 결과 분석
        print_results(results, output_file, test_size)

        return True

    except Exception as e:
        print(f"테스트 실행 오류: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        if engine:
            engine.cleanup()


def run_specific_id_test():
    """특정 ID 테스트 실행 (TEST_000 ~ TEST_007)"""
    
    target_ids = [f"TEST_{i:03d}" for i in range(8)]  # TEST_000 ~ TEST_007
    
    # 파일 존재 확인
    test_file = DEFAULT_FILES["test_file"]
    submission_file = DEFAULT_FILES["submission_file"]

    for file_path in [test_file, submission_file]:
        if not os.path.exists(file_path):
            print(f"오류: {file_path} 파일이 없습니다")
            return False

    engine = None
    try:
        # AI 엔진 초기화
        print("\n특정 ID 테스트 시스템 초기화 중...")
        engine = FinancialAIInference(verbose=False)

        # 테스트 데이터 준비
        import pandas as pd

        test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["encoding"])
        submission_df = pd.read_csv(
            submission_file, encoding=FILE_VALIDATION["encoding"]
        )

        # 특정 ID 필터링
        specific_test_df = test_df[test_df["ID"].isin(target_ids)].copy()
        specific_submission_df = submission_df[submission_df["ID"].isin(target_ids)].copy()

        if len(specific_test_df) == 0:
            print(f"오류: 지정된 ID 문항을 찾을 수 없습니다 ({', '.join(target_ids)})")
            print("실제 데이터의 첫 8개 문항으로 대체하여 테스트합니다.")
            
            # 처음 8개 문항으로 대체
            specific_test_df = test_df.head(8).copy()
            specific_submission_df = submission_df.head(8).copy()

        print(f"특정 ID 테스트 문항: {len(specific_test_df)}개")
        found_ids = specific_test_df["ID"].tolist()
        print(f"테스트할 문항 ID: {', '.join(found_ids)}")

        # 특정 ID 테스트 실행
        output_file = "./specific_id_test_result.csv"
        results = engine.execute_inference_with_data(
            specific_test_df, specific_submission_df, output_file
        )

        # 결과 출력
        print_specific_id_results(results, output_file, len(specific_test_df), found_ids)

        return True

    except Exception as e:
        print(f"특정 ID 테스트 실행 오류: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        if engine:
            engine.cleanup()


def run_question_type_test(question_type: str, test_size: int):
    """문항 유형별 테스트 실행"""

    # 파일 존재 확인
    test_file = DEFAULT_FILES["test_file"]
    submission_file = DEFAULT_FILES["submission_file"]

    for file_path in [test_file, submission_file]:
        if not os.path.exists(file_path):
            print(f"오류: {file_path} 파일이 없습니다")
            return False

    engine = None
    try:
        # AI 엔진 초기화
        print(f"\n{question_type} 테스트 시스템 초기화 중...")
        engine = FinancialAIInference(verbose=False)

        # 테스트 데이터 준비
        import pandas as pd

        test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["encoding"])
        submission_df = pd.read_csv(
            submission_file, encoding=FILE_VALIDATION["encoding"]
        )

        print(f"전체 데이터 분석 중: {len(test_df)}개 문항")

        # 해당 유형 문항 찾기
        type_indices = []
        type_questions = []
        
        print(f"{question_type} 문항 검색 중...")
        
        for idx, row in test_df.iterrows():
            question = row["Question"]
            question_id = row["ID"]
            
            # 질문 유형 분석
            detected_type, max_choice = engine.data_processor.extract_choice_range(question)
            
            if question_type == "주관식" and detected_type == "subjective":
                type_indices.append(idx)
                type_questions.append(question_id)
            elif question_type == "객관식" and detected_type == "multiple_choice":
                type_indices.append(idx)
                type_questions.append(question_id)
            
            # 원하는 문항 수만큼 찾으면 중단
            if len(type_indices) >= test_size:
                break
            
            # 진행률 표시 (50개마다)
            if (idx + 1) % 50 == 0:
                print(f"분석 진행: {idx + 1}/{len(test_df)} ({((idx + 1)/len(test_df)*100):.1f}%) - 찾은 {question_type} 문항: {len(type_indices)}개")

        if len(type_indices) == 0:
            print(f"오류: {question_type} 문항을 찾을 수 없습니다")
            
            if question_type == "주관식":
                print("모든 문항이 객관식으로 분류되었습니다.")
                print(f"테스트를 위해 처음 {test_size}개 문항을 주관식으로 처리합니다...")
                type_indices = list(range(min(test_size, len(test_df))))
                type_questions = test_df.iloc[type_indices]["ID"].tolist()
            else:
                print("모든 문항이 주관식으로 분류되었습니다.")
                print(f"테스트를 위해 처음 {test_size}개 문항을 객관식으로 처리합니다...")
                type_indices = list(range(min(test_size, len(test_df))))
                type_questions = test_df.iloc[type_indices]["ID"].tolist()

        # 찾은 문항 수 제한
        if len(type_indices) > test_size:
            type_indices = type_indices[:test_size]
            type_questions = type_questions[:test_size]

        print(f"\n{question_type} 문항 발견: {len(type_indices)}개")
        print(f"테스트할 문항 ID: {', '.join(type_questions[:10])}{'...' if len(type_questions) > 10 else ''}")

        # 해당 유형 데이터프레임 생성
        type_test_df = test_df.iloc[type_indices].copy()
        type_submission_df = submission_df.iloc[type_indices].copy()

        # 해당 유형 테스트 실행
        output_file = f"./{question_type}_test_result.csv"
        results = engine.execute_inference_with_data(
            type_test_df, type_submission_df, output_file
        )

        # 결과 출력
        if question_type == "주관식":
            print_subjective_results(results, output_file, len(type_indices), type_questions)
        else:
            print_multiple_choice_results(results, output_file, len(type_indices), type_questions)

        return True

    except Exception as e:
        print(f"{question_type} 테스트 실행 오류: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        if engine:
            engine.cleanup()


def print_specific_id_results(results: dict, output_file: str, test_count: int, question_ids: list):
    """특정 ID 테스트 결과 출력"""
    
    print(f"\n=== 특정 ID 테스트 완료 ({test_count}개 문항) ===")
    print(f"처리 시간: {results['total_time']:.1f}초")
    print(f"결과 파일: {output_file}")
    
    # 처리된 문항 ID 출력
    print(f"\n=== 처리된 문항 ID ===")
    print(f"총 {len(question_ids)}개 문항: {', '.join(question_ids)}")


def print_multiple_choice_results(results: dict, output_file: str, test_count: int, question_ids: list):
    """객관식 테스트 결과 출력"""
    
    print(f"\n=== 객관식 테스트 완료 ({test_count}개 문항) ===")
    print(f"처리 시간: {results['total_time']:.1f}초")
    print(f"결과 파일: {output_file}")
    
    # 처리된 문항 ID 출력
    print(f"\n=== 처리된 문항 ID ===")
    print(f"총 {len(question_ids)}개 문항: {', '.join(question_ids)}")


def print_subjective_results(results: dict, output_file: str, test_count: int, question_ids: list):
    """주관식 테스트 결과 출력"""
    
    print(f"\n=== 주관식 테스트 완료 ({test_count}개 문항) ===")
    print(f"처리 시간: {results['total_time']:.1f}초")
    print(f"결과 파일: {output_file}")
    
    # 실제 처리된 문항 ID 출력
    print(f"\n=== 처리된 문항 ID ===")
    print(f"총 {len(question_ids)}개 문항: {', '.join(question_ids)}")


def print_results(results: dict, output_file: str, test_size: int):
    """핵심 결과 출력"""

    total_time_minutes = results["total_time"] / 60
    print(f"\n=== 테스트 완료 ({test_size}개 문항) ===")
    print(f"처리 시간: {total_time_minutes:.1f}분")
    print(f"결과 파일: {output_file}")


def select_main_test_type():
    """메인 테스트 유형 선택"""
    print("\n=== AI 금융보안 테스트 시스템 ===")
    print("테스트할 방식을 선택하세요:")
    print()
    print("1. 객관식 테스트")
    print("2. 주관식 테스트")
    print("3. 특정 ID 테스트 (TEST_000 ~ TEST_007)")
    print()

    while True:
        try:
            choice = input("선택 (1-3): ").strip()

            if choice == "1":
                return "객관식"
            elif choice == "2":
                return "주관식"
            elif choice == "3":
                return "특정ID"
            else:
                print("잘못된 선택입니다. 1, 2, 3 중 하나를 입력하세요.")

        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            sys.exit(0)
        except Exception:
            print("잘못된 입력입니다. 다시 시도하세요.")


def select_question_count(test_type: str):
    """문항 수 선택"""
    print(f"\n{test_type} 테스트 문항 수를 선택하세요:")
    
    if test_type == "주관식":
        options = {
            "1": 1,
            "2": 2,
            "3": 4,
            "4": 10
        }
        print("1. 1문항")
        print("2. 2문항")
        print("3. 4문항")
        print("4. 10문항")
    else:  # 객관식
        options = {
            "1": 5,
            "2": 10,
            "3": 50,
            "4": 100
        }
        print("1. 5문항")
        print("2. 10문항")
        print("3. 50문항")
        print("4. 100문항")
    
    print()

    while True:
        try:
            choice = input("선택 (1-4): ").strip()

            if choice in options:
                return options[choice]
            else:
                print("잘못된 선택입니다. 1, 2, 3, 4 중 하나를 입력하세요.")

        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            sys.exit(0)
        except Exception:
            print("잘못된 입력입니다. 다시 시도하세요.")


def main():
    """메인 함수"""
    
    # 메인 테스트 유형 선택
    test_type = select_main_test_type()
    
    if test_type == "특정ID":
        print(f"\n특정 ID 테스트를 실행합니다...")
        print("TEST_000부터 TEST_007까지 8개 문항을 테스트합니다.")
        success = run_specific_id_test()
        if success:
            print(f"\n특정 ID 테스트 완료")
        else:
            print("\n특정 ID 테스트 실패")
            sys.exit(1)
    else:
        # 문항 수 선택
        question_count = select_question_count(test_type)
        
        print(f"\n{test_type} {question_count}문항 테스트를 실행합니다...")
        success = run_question_type_test(test_type, question_count)
        
        if success:
            print(f"\n{test_type} 테스트 완료")
        else:
            print(f"\n{test_type} 테스트 실패")
            sys.exit(1)


if __name__ == "__main__":
    main()