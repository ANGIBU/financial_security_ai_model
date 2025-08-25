# test_runner.py

import os
import sys
from pathlib import Path

current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from config import FILE_VALIDATION, DEFAULT_FILES
from inference import FinancialAIInference


def run_test(test_size: int = None, verbose: bool = True):
    """기본 테스트 실행"""

    if test_size is None:
        test_size = 50

    test_file = DEFAULT_FILES["test_file"]
    submission_file = DEFAULT_FILES["submission_file"]

    for file_path in [test_file, submission_file]:
        if not os.path.exists(file_path):
            print(f"오류: {file_path} 파일이 없습니다")
            return False

    engine = None
    try:
        engine = FinancialAIInference(verbose=False)

        import pandas as pd

        test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["encoding"])
        submission_df = pd.read_csv(
            submission_file, encoding=FILE_VALIDATION["encoding"]
        )

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

    except Exception as e:
        print(f"테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if engine:
            engine.cleanup()


def run_specific_id_test():
    """특정 ID 테스트 실행"""

    target_ids = [f"TEST_{i:03d}" for i in range(8)]

    test_file = DEFAULT_FILES["test_file"]
    submission_file = DEFAULT_FILES["submission_file"]

    for file_path in [test_file, submission_file]:
        if not os.path.exists(file_path):
            print(f"오류: {file_path} 파일이 없습니다")
            return False

    engine = None
    try:
        engine = FinancialAIInference(verbose=False)

        import pandas as pd

        test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["encoding"])
        submission_df = pd.read_csv(
            submission_file, encoding=FILE_VALIDATION["encoding"]
        )

        specific_test_df = test_df[test_df["ID"].isin(target_ids)].copy()
        specific_submission_df = submission_df[
            submission_df["ID"].isin(target_ids)
        ].copy()

        if len(specific_test_df) == 0:
            print(f"지정된 ID 문항을 찾을 수 없습니다. 첫 8개 문항으로 대체합니다.")
            specific_test_df = test_df.head(8).copy()
            specific_submission_df = submission_df.head(8).copy()

        found_ids = specific_test_df["ID"].tolist()

        output_file = "./specific_id_test_result.csv"
        results = engine.execute_inference_with_data(
            specific_test_df, specific_submission_df, output_file
        )

        print_specific_id_results(
            results, output_file, len(specific_test_df), found_ids
        )

        # 답변 분석 (엔진 재사용)
        print_answer_analysis(engine, specific_test_df, output_file)

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
    """질문 타입별 테스트 실행"""

    test_file = DEFAULT_FILES["test_file"]
    submission_file = DEFAULT_FILES["submission_file"]

    for file_path in [test_file, submission_file]:
        if not os.path.exists(file_path):
            print(f"오류: {file_path} 파일이 없습니다")
            return False

    engine = None
    try:
        engine = FinancialAIInference(verbose=False)

        import pandas as pd

        test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["encoding"])
        submission_df = pd.read_csv(
            submission_file, encoding=FILE_VALIDATION["encoding"]
        )

        print(f"데이터 로드 완료: {len(test_df)}개 문항")

        type_indices = []
        type_questions = []

        for idx, row in test_df.iterrows():
            question = row["Question"]
            question_id = row["ID"]

            detected_type, max_choice = engine.data_processor.extract_choice_range(
                question
            )

            if question_type == "주관식" and detected_type == "subjective":
                type_indices.append(idx)
                type_questions.append(question_id)
            elif question_type == "객관식" and detected_type == "multiple_choice":
                type_indices.append(idx)
                type_questions.append(question_id)

            if len(type_indices) >= test_size:
                break

        if len(type_indices) == 0:
            print(f"{question_type} 문항을 찾을 수 없어 처음 {test_size}개 문항으로 대체합니다.")
            type_indices = list(range(min(test_size, len(test_df))))
            type_questions = test_df.iloc[type_indices]["ID"].tolist()

        if len(type_indices) > test_size:
            type_indices = type_indices[:test_size]
            type_questions = type_questions[:test_size]

        type_test_df = test_df.iloc[type_indices].copy()
        type_submission_df = submission_df.iloc[type_indices].copy()

        output_file = f"./{question_type}_test_result.csv"
        results = engine.execute_inference_with_data(
            type_test_df, type_submission_df, output_file
        )

        if question_type == "주관식":
            print_subjective_results(
                results, output_file, len(type_indices), type_questions
            )
            # 주관식 상세 분석
            print_subjective_answer_analysis(engine, type_test_df, output_file)
        else:
            print_multiple_choice_results(
                results, output_file, len(type_indices), type_questions
            )

        return True

    except Exception as e:
        print(f"{question_type} 테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if engine:
            engine.cleanup()


def print_answer_analysis(engine, test_df, output_file):
    """답변 분석 (엔진 재사용)"""
    try:
        import pandas as pd
        result_df = pd.read_csv(output_file, encoding=FILE_VALIDATION["encoding"])
        
        print("\n=== 답변 분석 ===")
        
        subjective_count = 0
        objective_count = 0
        fallback_count = 0
        template_count = 0
        
        for idx, row in result_df.iterrows():
            question_id = row["ID"]
            answer = str(row["Answer"])
            
            question_row = test_df[test_df["ID"] == question_id]
            if len(question_row) > 0:
                question = question_row.iloc[0]["Question"]
                question_type, max_choice = engine.data_processor.extract_choice_range(question)
                
                if question_type == "subjective":
                    subjective_count += 1
                    
                    # 폴백 답변 패턴 체크
                    fallback_patterns = [
                        "관련 법령과 규정에 따라 체계적인 관리",
                        "체계적이고 전문적인 관리",
                        "지속적으로 운영해야 합니다"
                    ]
                    if any(pattern in answer for pattern in fallback_patterns):
                        fallback_count += 1
                    
                    # 템플릿 기반 답변 체크
                    template_indicators = [
                        "트로이 목마 기반 원격제어 악성코드",
                        "전자금융분쟁조정위원회",
                        "개인정보보호위원회",
                        "정보보안관리체계"
                    ]
                    if any(indicator in answer for indicator in template_indicators):
                        template_count += 1
                        
                else:
                    objective_count += 1
                    
        print(f"주관식: {subjective_count}개")
        print(f"객관식: {objective_count}개")
        
        if subjective_count > 0:
            fallback_rate = fallback_count / subjective_count * 100
            template_rate = template_count / subjective_count * 100
            success_rate = (subjective_count - fallback_count) / subjective_count * 100
            
            print(f"템플릿 기반 답변: {template_count}개 ({template_rate:.1f}%)")
            print(f"폴백 답변: {fallback_count}개 ({fallback_rate:.1f}%)")
            print(f"주관식 성공률: {success_rate:.1f}%")
        
        print("답변 분석 완료!")
        
    except Exception as e:
        print(f"답변 분석 중 오류: {e}")


def print_subjective_answer_analysis(engine, test_df, output_file):
    """주관식 답변 분석"""
    try:
        import pandas as pd
        result_df = pd.read_csv(output_file, encoding=FILE_VALIDATION["encoding"])
        
        print("\n=== 주관식 답변 상세 분석 ===")
        
        fallback_answers = []
        good_answers = []
        template_answers = []
        
        for idx, row in result_df.iterrows():
            question_id = row["ID"]
            answer = str(row["Answer"])
            
            # 폴백 패턴
            fallback_patterns = [
                "관련 법령과 규정에 따라 체계적인 관리가 필요합니다",
                "관련 법령과 규정에 따라 체계적인 관리를",
                "체계적인 관리가 필요합니다",
                "체계적이고 전문적인 관리",
                "지속적으로 운영해야 합니다"
            ]
            
            # 템플릿 기반 답변
            template_indicators = [
                "트로이 목마 기반 원격제어 악성코드(RAT)는",
                "전자금융분쟁조정위원회에서",
                "개인정보보호위원회에서 개인정보 보호에 관한",
                "정보보안관리체계를 구축하여"
            ]
            
            is_fallback = any(pattern in answer for pattern in fallback_patterns)
            is_template = any(indicator in answer for indicator in template_indicators)
            
            if is_fallback:
                fallback_answers.append(question_id)
            elif is_template:
                template_answers.append(question_id)
            else:
                good_answers.append(question_id)
                
        print(f"템플릿 기반 답변: {len(template_answers)}개")
        print(f"정상 LLM 답변: {len(good_answers)}개")
        print(f"폴백 답변: {len(fallback_answers)}개")
        
        if fallback_answers:
            print(f"폴백 답변 ID: {', '.join(fallback_answers[:5])}")
            
        total_answers = len(template_answers) + len(good_answers) + len(fallback_answers)
        if total_answers > 0:
            success_rate = (len(template_answers) + len(good_answers)) / total_answers * 100
            template_rate = len(template_answers) / total_answers * 100
            print(f"주관식 전체 성공률: {success_rate:.1f}%")
            print(f"템플릿 활용률: {template_rate:.1f}%")
        
        print("상세 분석 완료!")
        
    except Exception as e:
        print(f"주관식 답변 분석 중 오류: {e}")


def print_specific_id_results(
    results: dict, output_file: str, test_count: int, question_ids: list
):
    """특정 ID 테스트 결과 출력"""
    print(f"\n=== 특정 ID 테스트 완료 ({test_count}개 문항) ===")
    print(f"처리 시간: {results['total_time']:.1f}초")
    print(f"결과 파일: {output_file}")
    print(f"처리된 문항: {len(question_ids)}개")
    
    # 학습 데이터 정보
    if 'learning_data' in results:
        learning_data = results['learning_data']
        print(f"성공 답변 학습 데이터: {learning_data.get('successful_answers', 0)}개")
        print(f"실패 답변 학습 데이터: {learning_data.get('failed_answers', 0)}개")


def print_multiple_choice_results(
    results: dict, output_file: str, test_count: int, question_ids: list
):
    """객관식 테스트 결과 출력"""
    print(f"\n=== 객관식 테스트 완료 ({test_count}개 문항) ===")
    print(f"처리 시간: {results['total_time']:.1f}초")
    print(f"결과 파일: {output_file}")
    print(f"처리된 문항: {len(question_ids)}개")
    
    if 'domain_distribution' in results:
        print(f"도메인별 분포: {results['domain_distribution']}")
    if 'method_distribution' in results:
        print(f"방법별 분포: {results['method_distribution']}")


def print_subjective_results(
    results: dict, output_file: str, test_count: int, question_ids: list
):
    """주관식 테스트 결과 출력"""
    print(f"\n=== 주관식 테스트 완료 ({test_count}개 문항) ===")
    print(f"처리 시간: {results['total_time']:.1f}초")
    print(f"결과 파일: {output_file}")
    print(f"처리된 문항: {len(question_ids)}개")
    print(f"평균 처리 시간: {results.get('avg_processing_time', 0):.2f}초")
    
    if 'domain_distribution' in results:
        print(f"도메인별 분포: {results['domain_distribution']}")
    if 'learning_data' in results:
        learning_data = results['learning_data']
        print(f"성공 답변 학습: {learning_data.get('successful_answers', 0)}개")


def print_results(results: dict, output_file: str, test_size: int):
    """기본 결과 출력"""
    total_time_minutes = results["total_time"] / 60
    print(f"\n=== 테스트 완료 ({test_size}개 문항) ===")
    print(f"처리 시간: {total_time_minutes:.1f}분")
    print(f"결과 파일: {output_file}")
    
    # 성능 정보
    if 'avg_processing_time' in results:
        print(f"평균 문항 처리 시간: {results['avg_processing_time']:.2f}초")
    
    # 학습 데이터 정보
    if 'learning_data' in results:
        learning_data = results['learning_data']
        print(f"pkl 학습 데이터 저장:")
        print(f"  - 성공 답변: {learning_data.get('successful_answers', 0)}개")
        print(f"  - 실패 답변: {learning_data.get('failed_answers', 0)}개")
        print(f"  - 질문 패턴: {learning_data.get('question_patterns', 0)}개")


def select_main_test_type():
    """메인 테스트 타입 선택"""
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
    """질문 개수 선택"""
    print(f"\n{test_type} 테스트 문항 수를 선택하세요:")

    if test_type == "주관식":
        options = {"1": 1, "2": 2, "3": 4, "4": 10}
        print("1. 1문항")
        print("2. 2문항")
        print("3. 4문항")
        print("4. 10문항")
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
        except Exception:
            print("잘못된 입력입니다. 다시 시도하세요.")


def run_performance_monitoring_test(test_size: int = 20):
    """성능 모니터링 테스트"""
    print(f"\n=== 성능 모니터링 테스트 시작 ({test_size}개 문항) ===")
    
    test_file = DEFAULT_FILES["test_file"]
    submission_file = DEFAULT_FILES["submission_file"]

    for file_path in [test_file, submission_file]:
        if not os.path.exists(file_path):
            print(f"오류: {file_path} 파일이 없습니다")
            return False

    engine = None
    try:
        engine = FinancialAIInference(verbose=True)

        import pandas as pd
        import time

        test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["encoding"])
        submission_df = pd.read_csv(submission_file, encoding=FILE_VALIDATION["encoding"])

        # 샘플 추출
        if len(test_df) > test_size:
            test_df = test_df.head(test_size)
            submission_df = submission_df.head(test_size)

        print(f"모니터링 대상: {len(test_df)}개 문항")

        output_file = "./performance_monitoring_result.csv"
        
        # 시작 시간 기록
        start_time = time.time()
        
        results = engine.execute_inference_with_data(test_df, submission_df, output_file)
        
        end_time = time.time()
        total_time = end_time - start_time

        # 성능 분석
        print(f"\n=== 성능 분석 결과 ===")
        print(f"총 처리 시간: {total_time:.1f}초")
        print(f"평균 문항 처리 시간: {results.get('avg_processing_time', 0):.2f}초")
        print(f"도메인별 분포: {results.get('domain_distribution', {})}")
        print(f"방법별 분포: {results.get('method_distribution', {})}")
        
        # 학습 데이터 현황
        learning_data = results.get('learning_data', {})
        print(f"\n=== 학습 데이터 현황 ===")
        print(f"성공 답변 데이터: {learning_data.get('successful_answers', 0)}개")
        print(f"실패 답변 데이터: {learning_data.get('failed_answers', 0)}개")
        print(f"질문 패턴 데이터: {learning_data.get('question_patterns', 0)}개")

        # pkl 파일 확인
        from config import PKL_FILES
        print(f"\n=== pkl 파일 상태 ===")
        for file_type, file_path in PKL_FILES.items():
            if file_path.exists():
                file_size = file_path.stat().st_size
                print(f"{file_type}: {file_size} bytes")
            else:
                print(f"{file_type}: 파일 없음")

        # 로그 파일 확인
        from config import LOG_FILES
        print(f"\n=== 로그 파일 상태 ===")
        for log_type, log_path in LOG_FILES.items():
            if log_path.exists():
                log_size = log_path.stat().st_size
                print(f"{log_type}: {log_size} bytes")
            else:
                print(f"{log_type}: 파일 없음")

        print(f"\n성능 모니터링 테스트 완료!")
        return True

    except Exception as e:
        print(f"성능 모니터링 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if engine:
            engine.cleanup()


def main():
    """메인 함수"""

    test_type = select_main_test_type()

    if test_type == "특정ID":
        print(f"\n특정 ID 테스트를 실행합니다...")
        success = run_specific_id_test()
        if success:
            print(f"\n테스트 완료")
        else:
            print("\n테스트 실패")
            sys.exit(1)
    else:
        question_count = select_question_count(test_type)

        print(f"\n{test_type} {question_count}문항 테스트를 실행합니다...")
        success = run_question_type_test(test_type, question_count)

        if success:
            print(f"\n테스트 완료! 결과를 확인해주세요.")
            
            # 성능 모니터링 테스트 제안
            if question_count >= 10:
                print("\n성능 모니터링 테스트를 추가로 실행하시겠습니까? (y/n)")
                try:
                    choice = input().strip().lower()
                    if choice == 'y' or choice == 'yes':
                        run_performance_monitoring_test(min(question_count, 20))
                except:
                    pass
        else:
            print(f"\n테스트 실패")
            sys.exit(1)


if __name__ == "__main__":
    main()