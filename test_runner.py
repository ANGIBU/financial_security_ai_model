import os
import sys
from pathlib import Path
from datetime import datetime

current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from config import FILE_VALIDATION, DEFAULT_FILES, ensure_directories
from inference import FinancialAIInference


def run_test(test_size: int = None, verbose: bool = True):
    """기본 테스트 실행"""

    if test_size is None:
        test_size = 50

    try:
        ensure_directories()
        
        test_file = DEFAULT_FILES["test_file"]
        submission_file = DEFAULT_FILES["submission_file"]

        # 파일 존재 확인
        missing_files = []
        for name, file_path in [("test", test_file), ("submission", submission_file)]:
            if not Path(file_path).exists():
                missing_files.append(f"{name}: {file_path}")
        
        if missing_files:
            print("오류: 다음 파일들이 없습니다:")
            for missing in missing_files:
                print(f"  - {missing}")
            return False

        engine = None
        try:
            engine = FinancialAIInference(verbose=False, log_type="test")

            import pandas as pd

            # 데이터 로드
            try:
                test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["encoding"])
                submission_df = pd.read_csv(submission_file, encoding=FILE_VALIDATION["encoding"])
            except UnicodeDecodeError:
                print(f"인코딩 오류 발생, UTF-8으로 재시도...")
                test_df = pd.read_csv(test_file, encoding='utf-8')
                submission_df = pd.read_csv(submission_file, encoding='utf-8')
            except Exception as e:
                print(f"데이터 로드 실패: {e}")
                return False

            print(f"데이터 로드 완료: {len(test_df)}개 문항")

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

            # 간단 결과 출력
            print_results(results, output_file, test_size)

            return True

        except KeyboardInterrupt:
            print("\n테스트가 사용자에 의해 중단되었습니다.")
            return False
        except Exception as e:
            print(f"테스트 실행 오류: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            if engine:
                try:
                    engine.cleanup()
                except Exception as e:
                    print(f"엔진 정리 중 오류: {e}")

    except Exception as e:
        print(f"테스트 설정 오류: {e}")
        return False


def run_comprehensive_test(test_size: int):
    """종합 테스트 실행"""

    try:
        ensure_directories()
        
        test_file = DEFAULT_FILES["test_file"]
        submission_file = DEFAULT_FILES["submission_file"]

        # 파일 존재 확인
        missing_files = []
        for name, file_path in [("test", test_file), ("submission", submission_file)]:
            if not Path(file_path).exists():
                missing_files.append(f"{name}: {file_path}")
        
        if missing_files:
            print("오류: 다음 파일들이 없습니다:")
            for missing in missing_files:
                print(f"  - {missing}")
            return False

        engine = None
        try:
            engine = FinancialAIInference(verbose=False, log_type="test")

            import pandas as pd

            try:
                test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["encoding"])
                submission_df = pd.read_csv(submission_file, encoding=FILE_VALIDATION["encoding"])
            except UnicodeDecodeError:
                print(f"인코딩 오류 발생, UTF-8으로 재시도...")
                test_df = pd.read_csv(test_file, encoding='utf-8')
                submission_df = pd.read_csv(submission_file, encoding='utf-8')
            except Exception as e:
                print(f"데이터 로드 실패: {e}")
                return False

            print(f"데이터 로드 완료: {len(test_df)}개 문항")

            # 지정된 문항 수만큼 처리
            if len(test_df) > test_size:
                comprehensive_test_df = test_df.head(test_size).copy()
                comprehensive_submission_df = submission_df.head(test_size).copy()
            else:
                comprehensive_test_df = test_df.copy()
                comprehensive_submission_df = submission_df.copy()

            output_file = current_dir / "comprehensive_test_result.csv"
            results = engine.execute_inference_with_data(
                comprehensive_test_df, comprehensive_submission_df, output_file
            )

            # 간단 결과 출력
            print_comprehensive_results(results, output_file, len(comprehensive_test_df))

            return True

        except Exception as e:
            print(f"종합 테스트 실행 오류: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            if engine:
                try:
                    engine.cleanup()
                except Exception as e:
                    print(f"엔진 정리 중 오류: {e}")

    except Exception as e:
        print(f"종합 테스트 설정 오류: {e}")
        return False


def run_question_type_test(question_type: str, test_size: int):
    """질문 타입별 테스트 실행"""

    try:
        ensure_directories()
        
        test_file = DEFAULT_FILES["test_file"]
        submission_file = DEFAULT_FILES["submission_file"]

        # 파일 존재 확인
        missing_files = []
        for name, file_path in [("test", test_file), ("submission", submission_file)]:
            if not Path(file_path).exists():
                missing_files.append(f"{name}: {file_path}")
        
        if missing_files:
            print("오류: 다음 파일들이 없습니다:")
            for missing in missing_files:
                print(f"  - {missing}")
            return False

        engine = None
        try:
            engine = FinancialAIInference(verbose=False, log_type="test")

            import pandas as pd

            try:
                test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["encoding"])
                submission_df = pd.read_csv(submission_file, encoding=FILE_VALIDATION["encoding"])
            except UnicodeDecodeError:
                print(f"인코딩 오류 발생, UTF-8으로 재시도...")
                test_df = pd.read_csv(test_file, encoding='utf-8')
                submission_df = pd.read_csv(submission_file, encoding='utf-8')
            except Exception as e:
                print(f"데이터 로드 실패: {e}")
                return False

            print(f"데이터 로드 완료: {len(test_df)}개 문항")

            type_indices = []
            type_questions = []

            for idx, row in test_df.iterrows():
                question = row["Question"]
                question_id = row["ID"]

                try:
                    detected_type, max_choice = engine.data_processor.extract_choice_range(question)

                    if question_type == "주관식" and detected_type == "subjective":
                        type_indices.append(idx)
                        type_questions.append(question_id)
                    elif question_type == "객관식" and detected_type == "multiple_choice":
                        type_indices.append(idx)
                        type_questions.append(question_id)

                    if len(type_indices) >= test_size:
                        break
                except Exception as e:
                    print(f"질문 타입 분석 오류 ({question_id}): {e}")
                    continue

            if len(type_indices) == 0:
                print(f"{question_type} 문항을 찾을 수 없어 처음 {test_size}개 문항으로 대체합니다.")
                type_indices = list(range(min(test_size, len(test_df))))
                type_questions = test_df.iloc[type_indices]["ID"].tolist()

            if len(type_indices) > test_size:
                type_indices = type_indices[:test_size]
                type_questions = type_questions[:test_size]

            type_test_df = test_df.iloc[type_indices].copy()
            type_submission_df = submission_df.iloc[type_indices].copy()

            output_file = current_dir / f"{question_type}_test_result.csv"
            results = engine.execute_inference_with_data(
                type_test_df, type_submission_df, output_file
            )

            # 간단 결과 출력
            if question_type == "주관식":
                print_subjective_results(results, output_file, len(type_indices))
            else:
                print_multiple_choice_results(results, output_file, len(type_indices))

            return True

        except Exception as e:
            print(f"{question_type} 테스트 실행 오류: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            if engine:
                try:
                    engine.cleanup()
                except Exception as e:
                    print(f"엔진 정리 중 오류: {e}")

    except Exception as e:
        print(f"{question_type} 테스트 설정 오류: {e}")
        return False


def print_comprehensive_results(results: dict, output_file: Path, test_count: int):
    """종합 테스트 결과 출력"""
    try:
        print(f"결과 파일: {output_file.name}")
    except Exception as e:
        print(f"결과 파일: comprehensive_test_result.csv")


def print_multiple_choice_results(results: dict, output_file: Path, test_count: int):
    """객관식 테스트 결과 출력"""
    try:
        print(f"결과 파일: {output_file.name}")
    except Exception as e:
        print(f"결과 파일: 객관식_test_result.csv")


def print_subjective_results(results: dict, output_file: Path, test_count: int):
    """주관식 테스트 결과 출력"""
    try:
        print(f"결과 파일: {output_file.name}")
    except Exception as e:
        print(f"결과 파일: 주관식_test_result.csv")


def print_results(results: dict, output_file: str, test_size: int):
    """기본 결과 출력"""
    try:
        output_path = Path(output_file)
        print(f"결과 파일: {output_path.name}")
    except Exception as e:
        print(f"결과 파일: test_result.csv")


def select_main_test_type():
    """메인 테스트 타입 선택"""
    print("테스트할 방식을 선택하세요:")
    print()
    print("1. 객관식 테스트")
    print("2. 주관식 테스트")
    print("3. 종합 테스트")
    print()

    while True:
        try:
            choice = input("선택 (1-3): ").strip()

            if choice == "1":
                return "객관식"
            elif choice == "2":
                return "주관식"
            elif choice == "3":
                return "종합"
            else:
                print("잘못된 선택입니다. 1, 2, 3 중 하나를 입력하세요.")

        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            sys.exit(0)
        except Exception as e:
            print(f"입력 처리 오류: {e}. 다시 시도하세요.")


def select_question_count(test_type: str):
    """질문 개수 선택"""
    print(f"\n{test_type} 테스트 문항 수를 선택하세요:")

    if test_type == "주관식":
        options = {"1": 1, "2": 2, "3": 4, "4": 10}
        print("1. 1문항")
        print("2. 2문항")
        print("3. 4문항")
        print("4. 10문항")
    elif test_type == "종합":
        options = {"1": 5, "2": 8, "3": 50, "4": 100}
        print("1. 5문항")
        print("2. 8문항")
        print("3. 50문항")
        print("4. 100문항")
    else:
        options = {"1": 5, "2": 10, "3": 50, "4": 100}
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
        except Exception as e:
            print(f"입력 처리 오류: {e}. 다시 시도하세요.")


def main():
    """메인 함수"""

    try:
        test_type = select_main_test_type()

        if test_type == "종합":
            question_count = select_question_count(test_type)
            print(f"\n종합 {question_count}문항 테스트를 실행합니다...")
            success = run_comprehensive_test(question_count)
            if not success:
                sys.exit(1)
        else:
            question_count = select_question_count(test_type)
            print(f"\n{test_type} {question_count}문항 테스트를 실행합니다...")
            success = run_question_type_test(test_type, question_count)
            if not success:
                sys.exit(1)
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"메인 실행 오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()