# test_runner.py

import sys
from pathlib import Path

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
            domain_counts = {}

            for idx, row in test_df.iterrows():
                question = row["Question"]
                question_id = row["ID"]

                try:
                    detected_type, max_choice = engine.data_processor.extract_choice_range(question)
                    detected_domain = engine.data_processor.extract_domain(question)
                    
                    if detected_domain not in domain_counts:
                        domain_counts[detected_domain] = {"total": 0, "selected": 0}
                    domain_counts[detected_domain]["total"] += 1

                    if question_type == "주관식" and detected_type == "subjective":
                        type_indices.append(idx)
                        type_questions.append(question_id)
                        domain_counts[detected_domain]["selected"] += 1
                    elif question_type == "객관식" and detected_type == "multiple_choice":
                        type_indices.append(idx)
                        type_questions.append(question_id)
                        domain_counts[detected_domain]["selected"] += 1

                    if test_size != "전체" and len(type_indices) >= test_size:
                        break
                except Exception as e:
                    print(f"질문 타입 분석 오류 ({question_id}): {e}")
                    continue

            if len(type_indices) == 0:
                print(f"{question_type} 문항을 찾을 수 없어 처음 {test_size if test_size != '전체' else 50}개 문항으로 대체합니다.")
                type_indices = list(range(min(test_size if test_size != "전체" else 50, len(test_df))))
                type_questions = test_df.iloc[type_indices]["ID"].tolist()

            if test_size != "전체" and len(type_indices) > test_size:
                type_indices = type_indices[:test_size]
                type_questions = type_questions[:test_size]

            if domain_counts:
                print(f"\n{question_type} 문항 도메인 분포:")
                for domain, counts in domain_counts.items():
                    if counts["selected"] > 0:
                        percentage = (counts["selected"] / counts["total"]) * 100
                        print(f"  {domain}: {counts['selected']}/{counts['total']} ({percentage:.1f}%)")

            type_test_df = test_df.iloc[type_indices].copy()
            type_submission_df = submission_df.iloc[type_indices].copy()

            output_file = current_dir / f"{question_type}_test_result.csv"
            results = engine.execute_inference_with_data(
                type_test_df, type_submission_df, output_file
            )

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


def run_domain_test(domain_name: str, test_size: int):
    """도메인별 테스트 실행"""
    
    try:
        ensure_directories()
        
        test_file = DEFAULT_FILES["test_file"]
        submission_file = DEFAULT_FILES["submission_file"]

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

            domain_indices = []
            domain_questions = []
            question_type_counts = {"객관식": 0, "주관식": 0}

            for idx, row in test_df.iterrows():
                question = row["Question"]
                question_id = row["ID"]

                try:
                    detected_domain = engine.data_processor.extract_domain(question)
                    detected_type, max_choice = engine.data_processor.extract_choice_range(question)
                    
                    if detected_domain == domain_name:
                        domain_indices.append(idx)
                        domain_questions.append(question_id)
                        
                        if detected_type == "multiple_choice":
                            question_type_counts["객관식"] += 1
                        else:
                            question_type_counts["주관식"] += 1

                    if len(domain_indices) >= test_size:
                        break
                        
                except Exception as e:
                    print(f"도메인 분석 오류 ({question_id}): {e}")
                    continue

            if len(domain_indices) == 0:
                print(f"{domain_name} 도메인 문항을 찾을 수 없습니다.")
                return False

            if len(domain_indices) > test_size:
                domain_indices = domain_indices[:test_size]
                domain_questions = domain_questions[:test_size]

            print(f"\n{domain_name} 도메인 문항 유형 분포:")
            print(f"  객관식: {question_type_counts['객관식']}개")
            print(f"  주관식: {question_type_counts['주관식']}개")

            domain_test_df = test_df.iloc[domain_indices].copy()
            domain_submission_df = submission_df.iloc[domain_indices].copy()

            output_file = current_dir / f"{domain_name}_domain_test_result.csv"
            results = engine.execute_inference_with_data(
                domain_test_df, domain_submission_df, output_file
            )

            print_domain_results(results, output_file, len(domain_indices), domain_name)

            return True

        except Exception as e:
            print(f"{domain_name} 도메인 테스트 실행 오류: {e}")
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
        print(f"{domain_name} 도메인 테스트 설정 오류: {e}")
        return False


def print_comprehensive_results(results: dict, output_file: Path, test_count: int):
    """종합 테스트 결과 출력"""
    try:
        print(f"\n=== 종합 테스트 결과 ===")
        if results.get("success"):
            print(f"처리 완료: {test_count}개 문항")
            print(f"처리 시간: {results.get('total_time', 0):.1f}초")
            print(f"성공률: {results.get('success_rate', 0):.1f}%")
            
            if results.get('domain_performance'):
                print(f"\n도메인별 성능:")
                for domain, perf in results['domain_performance'].items():
                    success_rate = (perf['success'] / perf['total']) * 100 if perf['total'] > 0 else 0
                    print(f"  {domain}: {success_rate:.1f}% ({perf['success']}/{perf['total']})")
        
        print(f"결과 파일: {output_file.name}")
    except Exception as e:
        print(f"결과 파일: comprehensive_test_result.csv")


def print_multiple_choice_results(results: dict, output_file: Path, test_count: int):
    """객관식 테스트 결과 출력"""
    try:
        print(f"\n=== 객관식 테스트 결과 ===")
        if results.get("success"):
            print(f"처리 완료: {test_count}개 문항")
            print(f"처리 시간: {results.get('total_time', 0):.1f}초")
            print(f"성공률: {results.get('success_rate', 0):.1f}%")
        
        print(f"결과 파일: {output_file.name}")
    except Exception as e:
        print(f"결과 파일: 객관식_test_result.csv")


def print_subjective_results(results: dict, output_file: Path, test_count: int):
    """주관식 테스트 결과 출력"""
    try:
        print(f"\n=== 주관식 테스트 결과 ===")
        if results.get("success"):
            print(f"처리 완료: {test_count}개 문항")
            print(f"처리 시간: {results.get('total_time', 0):.1f}초")
            print(f"성공률: {results.get('success_rate', 0):.1f}%")
        
        print(f"결과 파일: {output_file.name}")
    except Exception as e:
        print(f"결과 파일: 주관식_test_result.csv")


def print_domain_results(results: dict, output_file: Path, test_count: int, domain_name: str):
    """도메인별 테스트 결과 출력"""
    try:
        print(f"\n=== {domain_name} 도메인 테스트 결과 ===")
        if results.get("success"):
            print(f"처리 완료: {test_count}개 문항")
            print(f"처리 시간: {results.get('total_time', 0):.1f}초")
            print(f"성공률: {results.get('success_rate', 0):.1f}%")
        
        print(f"결과 파일: {output_file.name}")
    except Exception as e:
        print(f"결과 파일: {domain_name}_domain_test_result.csv")


def print_results(results: dict, output_file: str, test_size: int):
    """기본 결과 출력"""
    try:
        print(f"\n=== 테스트 결과 ===")
        if results.get("success"):
            print(f"처리 완료: {test_size}개 문항")
            print(f"처리 시간: {results.get('total_time', 0):.1f}초")
            print(f"성공률: {results.get('success_rate', 0):.1f}%")
        
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
    print("4. 도메인별 테스트")
    print()

    while True:
        try:
            choice = input("선택 (1-4): ").strip()

            if choice == "1":
                return "객관식"
            elif choice == "2":
                return "주관식"
            elif choice == "3":
                return "종합"
            elif choice == "4":
                return "도메인별"
            else:
                print("잘못된 선택입니다. 1, 2, 3, 4 중 하나를 입력하세요.")

        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            sys.exit(0)
        except Exception as e:
            print(f"입력 처리 오류: {e}. 다시 시도하세요.")


def select_domain():
    """도메인 선택"""
    domains = [
        "사이버보안",
        "전자금융", 
        "개인정보보호",
        "정보보안",
        "금융투자",
        "위험관리"
    ]
    
    print("\n테스트할 도메인을 선택하세요:")
    for i, domain in enumerate(domains, 1):
        print(f"{i}. {domain}")
    print()
    
    while True:
        try:
            choice = input(f"선택 (1-{len(domains)}): ").strip()
            
            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(domains):
                    return domains[choice_num - 1]
            
            print(f"잘못된 선택입니다. 1~{len(domains)} 중 하나를 입력하세요.")
            
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            sys.exit(0)
        except Exception as e:
            print(f"입력 처리 오류: {e}. 다시 시도하세요.")


def select_question_count(test_type: str):
    """질문 개수 선택"""
    print(f"\n{test_type} 테스트 문항 수를 선택하세요:")

    if test_type == "주관식":
        options = {"1": 1, "2": 2, "3": 4, "4": 10, "5": "전체"}
        print("1. 1문항")
        print("2. 2문항")
        print("3. 4문항")
        print("4. 10문항")
        print("5. 전체 테스트")
    elif test_type == "객관식":
        options = {"1": 5, "2": 10, "3": 50, "4": 100, "5": "전체"}
        print("1. 5문항")
        print("2. 10문항")
        print("3. 50문항")
        print("4. 100문항")
        print("5. 전체 테스트")
    elif test_type == "종합":
        options = {"1": 5, "2": 8, "3": 50, "4": 100}
        print("1. 5문항")
        print("2. 8문항")
        print("3. 50문항")
        print("4. 100문항")
    elif test_type == "도메인별":
        options = {"1": 5, "2": 10, "3": 20, "4": 50}
        print("1. 5문항")
        print("2. 10문항")
        print("3. 20문항")
        print("4. 50문항")
    else:
        options = {"1": 5, "2": 10, "3": 50, "4": 100}
        print("1. 5문항")
        print("2. 10문항")
        print("3. 50문항")
        print("4. 100문항")

    print()

    while True:
        try:
            choice = input("선택 (1-5): ").strip()

            if choice in options:
                return options[choice]
            else:
                if test_type in ["주관식", "객관식"]:
                    print("잘못된 선택입니다. 1, 2, 3, 4, 5 중 하나를 입력하세요.")
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
        elif test_type == "도메인별":
            domain = select_domain()
            question_count = select_question_count(test_type)
            print(f"\n{domain} 도메인 {question_count}문항 테스트를 실행합니다...")
            success = run_domain_test(domain, question_count)
            if not success:
                sys.exit(1)
        else:
            question_count = select_question_count(test_type)
            if question_count == "전체":
                print(f"\n{test_type} 전체 테스트를 실행합니다...")
            else:
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