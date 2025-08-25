# test_runner.py

import os
import sys
from pathlib import Path

current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from config import FILE_VALIDATION, DEFAULT_FILES
from inference import FinancialAIInference


def run_test(test_size: int = None, verbose: bool = True):

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

        # 답변 분석 - 학습 통계 포함
        print_answer_analysis_with_learning_stats(engine, specific_test_df, output_file)

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
            # 주관식 상세 분석 - 학습 통계 포함
            print_subjective_answer_analysis_with_learning_stats(engine, type_test_df, output_file)
        else:
            print_multiple_choice_results(
                results, output_file, len(type_indices), type_questions
            )
            # 객관식 분석 - 학습 통계 포함
            print_mc_answer_analysis_with_learning_stats(engine, type_test_df, output_file)

        return True

    except Exception as e:
        print(f"{question_type} 테스트 실행 오류: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        if engine:
            engine.cleanup()


def print_answer_analysis_with_learning_stats(engine, test_df, output_file):
    """답변 분석 - 학습 통계 포함"""
    try:
        import pandas as pd
        result_df = pd.read_csv(output_file, encoding=FILE_VALIDATION["encoding"])
        
        print("\n=== 답변 분석 완료 ===")
        
        # 학습 통계를 파일로 로그
        engine.learning_manager.log_learning_stats("답변 분석")
        
        # 현재 테스트 결과 분석
        subjective_count = 0
        objective_count = 0
        fallback_count = 0
        
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
                else:
                    objective_count += 1
        
        # 결과를 파일로 로그
        engine.learning_manager.log_to_file(f"테스트 결과 - 주관식: {subjective_count}개, 객관식: {objective_count}개")
        if subjective_count > 0:
            engine.learning_manager.log_to_file(f"폴백 답변: {fallback_count}개 ({fallback_count/subjective_count*100:.1f}%)")
            
    except Exception as e:
        print(f"답변 분석 중 오류: {e}")


def print_subjective_answer_analysis_with_learning_stats(engine, test_df, output_file):
    """주관식 답변 상세 분석 - 학습 통계 포함"""
    try:
        import pandas as pd
        result_df = pd.read_csv(output_file, encoding=FILE_VALIDATION["encoding"])
        
        print("\n=== 주관식 답변 상세 분석 ===")
        
        # 학습 통계를 파일로 로그
        engine.learning_manager.log_learning_stats("주관식 분석")
        
        template_answers = []
        llm_answers = []
        fallback_answers = []
        
        for idx, row in result_df.iterrows():
            question_id = row["ID"]
            answer = str(row["Answer"])
            
            # 특화 템플릿 답변인지 확인
            specialized_indicators = [
                "트로이 목마 기반 원격제어 악성코드",
                "전자금융분쟁조정위원회",
                "개인정보보호위원회",
                "개인정보침해신고센터"
            ]
            
            is_specialized = any(indicator in answer for indicator in specialized_indicators)
            
            fallback_patterns = [
                "관련 법령과 규정에 따라 체계적인 관리가 필요합니다",
                "관련 법령과 규정에 따라 체계적인 관리를",
                "체계적인 관리가 필요합니다",
                "체계적이고 전문적인 관리",
                "지속적으로 운영해야 합니다"
            ]
            
            is_fallback = any(pattern in answer for pattern in fallback_patterns)
            
            if is_fallback:
                fallback_answers.append(question_id)
            elif is_specialized:
                template_answers.append(question_id)
            else:
                llm_answers.append(question_id)
        
        total_answers = len(template_answers) + len(llm_answers) + len(fallback_answers)
        
        # 결과를 파일로 로그
        engine.learning_manager.log_to_file(f"답변 유형 분석:")
        engine.learning_manager.log_to_file(f"  특화 템플릿: {len(template_answers)}개 ({len(template_answers)/total_answers*100:.1f}%)")
        engine.learning_manager.log_to_file(f"  LLM 생성: {len(llm_answers)}개 ({len(llm_answers)/total_answers*100:.1f}%)")
        engine.learning_manager.log_to_file(f"  폴백 답변: {len(fallback_answers)}개 ({len(fallback_answers)/total_answers*100:.1f}%)")
        
        if fallback_answers:
            engine.learning_manager.log_to_file(f"폴백 답변 ID: {', '.join(fallback_answers[:5])}{' 등' if len(fallback_answers) > 5 else ''}")
        
        # 성공률 계산
        success_answers = len(template_answers) + len(llm_answers)
        success_rate = success_answers / total_answers * 100 if total_answers > 0 else 0
        engine.learning_manager.log_to_file(f"주관식 성공률: {success_rate:.1f}%")
        print(f"주관식 성공률: {success_rate:.1f}%")
        
        # 도메인별 분석
        domain_counts = {}
        for idx, row in result_df.iterrows():
            question_id = row["ID"]
            question_row = test_df[test_df["ID"] == question_id]
            if len(question_row) > 0:
                question = question_row.iloc[0]["Question"]
                domain = engine.data_processor.extract_domain(question)
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        if domain_counts:
            engine.learning_manager.log_to_file(f"도메인별 문항 수:")
            for domain, count in domain_counts.items():
                engine.learning_manager.log_to_file(f"  {domain}: {count}개")
        
        print("분석 완료!")
        
    except Exception as e:
        print(f"주관식 답변 분석 중 오류: {e}")


def print_mc_answer_analysis_with_learning_stats(engine, test_df, output_file):
    """객관식 답변 분석 - 학습 통계 포함"""
    try:
        import pandas as pd
        result_df = pd.read_csv(output_file, encoding=FILE_VALIDATION["encoding"])
        
        print("\n=== 객관식 답변 분석 ===")
        
        # 학습 통계를 파일로 로그
        mc_patterns = engine.learning_manager.learning_data.get("mc_patterns", {})
        if mc_patterns:
            engine.learning_manager.log_to_file(f"학습된 객관식 패턴: {len(mc_patterns)}개")
            for pattern, data in list(mc_patterns.items())[:3]:
                engine.learning_manager.log_to_file(f"  {pattern}: {data['count']}회 학습")
        
        # 답변 분포 분석
        answer_distribution = {}
        pattern_matches = 0
        
        for idx, row in result_df.iterrows():
            question_id = row["ID"]
            answer = str(row["Answer"])
            
            if answer.isdigit():
                answer_distribution[answer] = answer_distribution.get(answer, 0) + 1
            
            # 패턴 매칭 확인
            question_row = test_df[test_df["ID"] == question_id]
            if len(question_row) > 0:
                question = question_row.iloc[0]["Question"]
                learned_prediction = engine.learning_manager.get_mc_prediction(question, 5)
                if learned_prediction and learned_prediction == answer:
                    pattern_matches += 1
        
        # 결과를 파일로 로그
        engine.learning_manager.log_to_file(f"답변 분포:")
        for answer_num in sorted(answer_distribution.keys()):
            count = answer_distribution[answer_num]
            percentage = count / len(result_df) * 100
            engine.learning_manager.log_to_file(f"  {answer_num}번: {count}개 ({percentage:.1f}%)")
        
        if len(result_df) > 0:
            match_rate = pattern_matches/len(result_df)*100
            engine.learning_manager.log_to_file(f"학습 패턴 매칭률: {match_rate:.1f}%")
            print(f"학습 패턴 매칭률: {match_rate:.1f}%")
        
        print("객관식 분석 완료!")
        
    except Exception as e:
        print(f"객관식 답변 분석 중 오류: {e}")


def print_specific_id_results(
    results: dict, output_file: str, test_count: int, question_ids: list
):
    print(f"\n=== 특정 ID 테스트 완료 ({test_count}개 문항) ===")
    print(f"처리 시간: {results['total_time']:.1f}초")
    print(f"결과 파일: {output_file}")
    print(f"처리된 문항: {len(question_ids)}개")


def print_multiple_choice_results(
    results: dict, output_file: str, test_count: int, question_ids: list
):
    print(f"\n=== 객관식 테스트 완료 ({test_count}개 문항) ===")
    print(f"처리 시간: {results['total_time']:.1f}초")
    print(f"결과 파일: {output_file}")
    print(f"처리된 문항: {len(question_ids)}개")


def print_subjective_results(
    results: dict, output_file: str, test_count: int, question_ids: list
):
    print(f"\n=== 주관식 테스트 완료 ({test_count}개 문항) ===")
    print(f"처리 시간: {results['total_time']:.1f}초")
    print(f"결과 파일: {output_file}")
    print(f"처리된 문항: {len(question_ids)}개")


def print_results(results: dict, output_file: str, test_size: int):
    total_time_minutes = results["total_time"] / 60
    print(f"\n=== 테스트 완료 ({test_size}개 문항) ===")
    print(f"처리 시간: {total_time_minutes:.1f}분")
    print(f"결과 파일: {output_file}")
    
    if "learning_stats" in results:
        print("학습 통계가 log 폴더의 분석 리포트에 저장되었습니다.")


def select_main_test_type():
    print("테스트할 방식을 선택하세요:")
    print()
    print("1. 객관식 테스트")
    print("2. 주관식 테스트")
    print("3. 특정 ID 테스트 (TEST_000 ~ TEST_007)")
    print("4. 학습 통계 확인")
    print()

    while True:
        try:
            choice = input("선택 (1-4): ").strip()

            if choice == "1":
                return "객관식"
            elif choice == "2":
                return "주관식"
            elif choice == "3":
                return "특정ID"
            elif choice == "4":
                return "학습통계"
            else:
                print("잘못된 선택입니다. 1, 2, 3, 4 중 하나를 입력하세요.")

        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            sys.exit(0)
        except Exception:
            print("잘못된 입력입니다. 다시 시도하세요.")


def show_learning_statistics():
    """학습 통계 표시"""
    try:
        engine = FinancialAIInference(verbose=False)
        
        print("\n=== 현재 학습 통계 ===")
        
        # 모든 통계를 파일로 로그
        engine.learning_manager.log_learning_stats("학습 통계 조회")
        
        # 객관식 패턴 학습 현황을 파일로 로그
        mc_patterns = engine.learning_manager.learning_data.get("mc_patterns", {})
        if mc_patterns:
            engine.learning_manager.log_to_file("학습된 객관식 패턴:")
            for pattern, data in mc_patterns.items():
                engine.learning_manager.log_to_file(f"  {pattern}: {data['count']}회 학습")
        else:
            engine.learning_manager.log_to_file("학습된 객관식 패턴이 없습니다.")
        
        # 학습 분석 리포트 생성
        engine.learning_manager.export_analysis()
        print("학습 통계가 log 폴더의 분석 리포트에 저장되었습니다.")
        
        engine.cleanup()
        
    except Exception as e:
        print(f"학습 통계 확인 중 오류: {e}")


def select_question_count(test_type: str):
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


def main():

    test_type = select_main_test_type()

    if test_type == "학습통계":
        show_learning_statistics()
        return
    elif test_type == "특정ID":
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
            print(f"\n✅ {test_type} 테스트 완료! 프로그램을 종료합니다.")
        else:
            print(f"\n❌ {test_type} 테스트 실패")
            sys.exit(1)


if __name__ == "__main__":
    main()