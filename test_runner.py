# test_runner.py

"""
테스트 실행기
- 시스템 테스트 실행
- 의도 일치 성공률 표시
- 핵심 성능 지표 출력
"""

import os
import sys
from pathlib import Path

# 현재 디렉토리 설정
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# 설정 파일 import
from config import TEST_CONFIG, FILE_VALIDATION, DEFAULT_FILES
from inference import FinancialAIInference


def run_test(test_size: int = None, verbose: bool = True):
    """테스트 실행"""

    # 기본 테스트 크기 설정
    if test_size is None:
        test_size = TEST_CONFIG["default_test_size"]

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
        print_enhanced_results(results, output_file, test_size)

        return True

    except Exception as e:
        print(f"테스트 실행 오류: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        if engine:
            engine.cleanup()


def run_subjective_test():
    """주관식 테스트 실행"""

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
        print("\n주관식 테스트 시스템 초기화 중...")
        engine = FinancialAIInference(verbose=False)

        # 테스트 데이터 준비
        import pandas as pd

        test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["encoding"])
        submission_df = pd.read_csv(
            submission_file, encoding=FILE_VALIDATION["encoding"]
        )

        print(f"전체 데이터 분석 중: {len(test_df)}개 문항")

        # 주관식 문항 동적으로 찾기
        subjective_indices = []
        subjective_questions = []
        
        for idx, row in test_df.iterrows():
            question = row["Question"]
            question_id = row["ID"]
            
            # 질문 유형 분석
            question_type, max_choice = engine.data_processor.extract_choice_range(question)
            
            if question_type == "subjective":
                subjective_indices.append(idx)
                subjective_questions.append(question_id)
            
            # 처리 진행률 표시
            if (idx + 1) % 50 == 0:
                print(f"분석 진행: {idx + 1}/{len(test_df)} ({((idx + 1)/len(test_df)*100):.1f}%)")

        if len(subjective_indices) == 0:
            print(f"오류: 주관식 문항을 찾을 수 없습니다")
            print("모든 문항이 객관식으로 분류되었습니다.")
            
            # 강제로 몇 개 문항을 주관식으로 처리 (테스트 목적)
            print("테스트를 위해 처음 5개 문항을 주관식으로 처리합니다...")
            subjective_indices = list(range(min(5, len(test_df))))
            subjective_questions = test_df.iloc[subjective_indices]["ID"].tolist()

        # 주관식 문항이 너무 많으면 일부만 선택
        max_subjective_test = 10
        if len(subjective_indices) > max_subjective_test:
            print(f"주관식 문항이 {len(subjective_indices)}개로 많아서 처음 {max_subjective_test}개만 테스트합니다.")
            subjective_indices = subjective_indices[:max_subjective_test]
            subjective_questions = subjective_questions[:max_subjective_test]

        print(f"주관식 문항 발견: {len(subjective_indices)}개")
        print(f"테스트할 문항 ID: {', '.join(subjective_questions[:5])}{'...' if len(subjective_questions) > 5 else ''}")

        # 주관식 데이터프레임 생성
        subjective_test_df = test_df.iloc[subjective_indices].copy()
        subjective_submission_df = submission_df.iloc[subjective_indices].copy()

        # 주관식 테스트 실행
        output_file = "./subjective_test_result.csv"
        results = engine.execute_inference_with_data(
            subjective_test_df, subjective_submission_df, output_file
        )

        # 결과 출력
        print_subjective_results(results, output_file, len(subjective_indices), subjective_questions)

        return True

    except Exception as e:
        print(f"주관식 테스트 실행 오류: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        if engine:
            engine.cleanup()


def print_subjective_results(results: dict, output_file: str, test_count: int, question_ids: list):
    """주관식 테스트 결과 출력"""
    
    print(f"\n=== 주관식 테스트 완료 ({test_count}개 문항) ===")
    print(f"처리 시간: {results['total_time']:.1f}초")
    print(f"결과 파일: {output_file}")
    
    # 주관식 성능 지표
    subj_count = results.get("subj_count", 0)
    intent_success_rate = results.get("intent_match_success_rate", 0)
    korean_compliance = results.get("korean_compliance_rate", 0)
    avg_quality = results.get("avg_quality_score", 0)
    llm_usage = results.get("llm_usage_rate", 0)
    
    print(f"\n=== 주관식 성능 분석 ===")
    print(f"처리된 주관식 문항: {subj_count}개")
    print(f"의도 일치 성공률: {intent_success_rate:.1f}%")
    print(f"한국어 준수율: {korean_compliance:.1f}%")
    print(f"평균 품질 점수: {avg_quality:.2f}/1.0")
    print(f"LLM 활용률: {llm_usage:.1f}%")
    
    # 특화 기능 성능
    institution_count = results.get("institution_questions_count", 0)
    template_usage = results.get("template_usage_rate", 0)
    text_recovery = results.get("text_recovery_rate", 0)
    
    print(f"\n=== 특화 기능 성능 ===")
    print(f"기관 질문 처리: {institution_count}개")
    print(f"템플릿 활용률: {template_usage:.1f}%")
    print(f"텍스트 복구율: {text_recovery:.1f}%")
    
    # 품질 개선 통계
    quality_improvements = results.get("quality_improvement_count", 0)
    korean_enhancements = results.get("korean_enhancement_count", 0)
    grammar_fixes = results.get("grammar_fix_rate", 0)
    
    print(f"\n=== 품질 개선 통계 ===")
    print(f"품질 개선 횟수: {quality_improvements}회")
    print(f"한국어 강화 횟수: {korean_enhancements}회")
    print(f"문법 수정률: {grammar_fixes:.1f}%")
    
    # 도메인별 분석
    domain_stats = results.get("domain_stats", {})
    if domain_stats:
        print(f"\n=== 도메인별 분포 ===")
        for domain, count in domain_stats.items():
            percentage = (count / test_count) * 100
            print(f"{domain}: {count}개 ({percentage:.1f}%)")
    
    # 오류 분석
    validation_errors = results.get("validation_error_rate", 0)
    if validation_errors > 0:
        print(f"\n=== 오류 분석 ===")
        print(f"검증 실패율: {validation_errors:.1f}%")
    
    # 실제 처리된 문항 ID 출력
    print(f"\n=== 처리된 문항 ID ===")
    print(f"총 {len(question_ids)}개 문항: {', '.join(question_ids)}")


def print_enhanced_results(results: dict, output_file: str, test_size: int):
    """핵심 성과 지표 출력"""

    total_time_minutes = results["total_time"] / 60
    print(f"\n=== 테스트 완료 ({test_size}개 문항) ===")
    print(f"처리 시간: {total_time_minutes:.1f}분")
    print(f"처리 문항: {results['total_questions']}개")

    # 객관식 성능
    mc_count = results.get("mc_count", 0)
    mc_success_rate = results.get("model_success_rate", 0)
    mc_context_accuracy = results.get("mc_context_accuracy_rate", 0)
    mc_pattern_match = results.get("mc_pattern_match_rate", 0)

    print(
        f"\n=== 객관식 성능 (전체의 {(mc_count/results['total_questions']*100):.0f}%) ==="
    )
    print(f"객관식 문항: {mc_count}개")
    print(f"기본 성공률: {mc_success_rate:.1f}%")
    print(f"컨텍스트 정확도: {mc_context_accuracy:.1f}%")
    print(f"패턴 매칭률: {mc_pattern_match:.1f}%")

    # 도메인별 객관식 성과
    mc_domain_rates = results.get("mc_domain_accuracy_rates", {})
    if mc_domain_rates:
        print("도메인별 객관식 정확도:")
        for domain, rate in mc_domain_rates.items():
            print(f"  - {domain}: {rate:.1f}%")

    # 주관식 성능
    subj_count = results.get("subj_count", 0)
    intent_success_rate = results.get("intent_match_success_rate", 0)
    korean_compliance = results.get("korean_compliance_rate", 0)
    avg_quality = results.get("avg_quality_score", 0)

    print(
        f"\n=== 주관식 성능 (전체의 {(subj_count/results['total_questions']*100):.0f}%) ==="
    )
    print(f"주관식 문항: {subj_count}개")
    if subj_count > 0:
        print(f"의도 일치율: {intent_success_rate:.1f}%")
        print(f"한국어 준수율: {korean_compliance:.1f}%")
        print(f"평균 품질점수: {avg_quality:.2f}/1.0")
    else:
        print("주관식 문항 없음")

    # 특화 기능 성능
    institution_count = results.get("institution_questions_count", 0)
    institution_accuracy = results.get("institution_answer_accuracy", 0)
    template_usage = results.get("template_usage_rate", 0)

    print(f"\n=== 특화 기능 성능 ===")
    print(f"기관 질문 처리: {institution_count}개")
    if institution_count > 0:
        print(f"기관 답변 정확도: {institution_accuracy}개 성공")
    print(f"템플릿 활용률: {template_usage:.1f}%")

    # 최적화 성과
    quality_improvements = results.get("quality_improvement_count", 0)
    fallback_avoidance = results.get("fallback_avoidance_rate", 0)
    korean_enhancements = results.get("korean_enhancement_count", 0)

    print(f"\n=== 최적화 성과 ===")
    print(f"품질 개선: {quality_improvements}회")
    print(f"폴백 회피율: {fallback_avoidance:.1f}%")
    print(f"한국어 강화: {korean_enhancements}회")

    # 오류 분석
    choice_errors = results.get("choice_range_error_rate", 0)
    validation_errors = results.get("validation_error_rate", 0)

    if choice_errors > 0 or validation_errors > 0:
        print(f"\n=== 오류 분석 ===")
        if choice_errors > 0:
            print(f"선택지 범위 오류율: {choice_errors:.1f}%")
        if validation_errors > 0:
            print(f"검증 실패율: {validation_errors:.1f}%")

    # 도메인별 성과 요약
    domain_stats = results.get("domain_stats", {})
    if domain_stats:
        print(f"\n=== 도메인별 분포 ===")
        for domain, count in domain_stats.items():
            percentage = (count / results["total_questions"]) * 100
            print(f"{domain}: {count}개 ({percentage:.1f}%)")


def estimate_final_performance(results: dict) -> float:
    """최종 성능 예측"""

    # 객관식 성과 (가중치 86%)
    mc_weight = 0.86
    mc_success = results.get("model_success_rate", 0) / 100
    mc_context = results.get("mc_context_accuracy_rate", 0) / 100
    mc_pattern = results.get("mc_pattern_match_rate", 0) / 100

    mc_score = (mc_success * 0.4 + mc_context * 0.3 + mc_pattern * 0.3) * mc_weight

    # 주관식 성과 (가중치 14%)
    subj_weight = 0.14
    intent_success = results.get("intent_match_success_rate", 0) / 100
    korean_compliance = results.get("korean_compliance_rate", 0) / 100
    quality_score = results.get("avg_quality_score", 0)

    subj_score = (
        intent_success * 0.4 + korean_compliance * 0.3 + quality_score * 0.3
    ) * subj_weight

    # 전체 예상 점수
    total_score = mc_score + subj_score

    return total_score


def suggest_improvements(results: dict):
    """개선 제안"""
    print(f"\n=== 성능 개선 제안 ===")

    mc_context = results.get("mc_context_accuracy_rate", 0)
    intent_rate = results.get("intent_match_success_rate", 0)
    model_rate = results.get("model_success_rate", 0)
    validation_error = results.get("validation_error_rate", 0)
    choice_error = results.get("choice_range_error_rate", 0)

    improvements = []

    if mc_context < 85:
        improvements.append(
            f"객관식 컨텍스트 정확도가 {mc_context:.1f}%로 낮습니다. 도메인별 패턴 학습을 강화하세요."
        )

    if intent_rate < 70:
        improvements.append(
            f"의도 일치 성공률이 {intent_rate:.1f}%로 낮습니다. 질문 의도 분석 정확도를 높이세요."
        )

    if model_rate < 90:
        improvements.append(
            f"모델 성공률이 {model_rate:.1f}%로 낮습니다. 프롬프트 최적화가 필요합니다."
        )

    if validation_error > 5:
        improvements.append(
            f"검증 오류율이 {validation_error:.1f}%입니다. 답변 검증 로직을 개선하세요."
        )

    if choice_error > 2:
        improvements.append(
            f"선택지 범위 오류율이 {choice_error:.1f}%입니다. 객관식 답변 처리를 개선하세요."
        )

    if not improvements:
        improvements.append("현재 성능이 우수합니다. 지속적인 모니터링을 유지하세요.")

    for i, improvement in enumerate(improvements, 1):
        print(f"{i}. {improvement}")


def analyze_domain_performance(results: dict):
    """도메인별 성능 분석"""
    print(f"\n=== 도메인별 성능 분석 ===")

    domain_stats = results.get("domain_stats", {})
    mc_domain_rates = results.get("mc_domain_accuracy_rates", {})
    domain_intent_rates = results.get("domain_intent_match_rates", {})

    if not domain_stats:
        print("도메인 분석 데이터가 없습니다.")
        return

    for domain, count in domain_stats.items():
        print(f"\n[{domain}] - {count}개 문항")

        if domain in mc_domain_rates:
            print(f"  객관식 정확도: {mc_domain_rates[domain]:.1f}%")

        if domain in domain_intent_rates:
            print(f"  의도 일치율: {domain_intent_rates[domain]:.1f}%")

        # 도메인별 권장사항
        if domain == "사이버보안":
            print(f"  권장: 트로이 목마, RAT, SBOM 관련 패턴 강화")
        elif domain == "개인정보보호":
            print(f"  권장: 기관명 답변과 법정대리인 관련 답변 정확도 개선")
        elif domain == "전자금융":
            print(f"  권장: 분쟁조정위원회 관련 기관 답변 정확도 향상")
        elif domain == "위험관리":
            print(f"  권장: 부정형 질문 패턴 인식 개선")
        elif domain == "금융투자":
            print(f"  권장: 업무 구분 관련 객관식 정확도 향상")


def select_test_size():
    """테스트 문항 수 선택"""
    print("\n테스트할 문항 수를 선택하세요:")

    test_options = TEST_CONFIG["test_sizes"]
    print(f"1. {test_options['mini']}문항 (미니 테스트)")
    print(f"2. {test_options['basic']}문항 (기본 테스트)")
    print(f"3. {test_options['detailed']}문항 (정밀 테스트)")
    print(f"4. 주관식 테스트 (동적으로 주관식 문항 검색)")
    print()

    while True:
        try:
            choice = input("선택 (1-4): ").strip()

            if choice == "1":
                return test_options["mini"]
            elif choice == "2":
                return test_options["basic"]
            elif choice == "3":
                return test_options["detailed"]
            elif choice == "4":
                return "subjective"
            else:
                print("잘못된 선택입니다. 1, 2, 3, 4 중 하나를 입력하세요.")

        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            sys.exit(0)
        except Exception:
            print("잘못된 입력입니다. 다시 시도하세요.")


def main():
    """메인 함수"""
    # 테스트 크기 선택
    test_size = select_test_size()

    if test_size == "subjective":
        print(f"\n주관식 테스트를 실행합니다...")
        print("전체 데이터에서 주관식 문항을 자동으로 찾아서 테스트합니다.")
        success = run_subjective_test()
        if success:
            print(f"\n주관식 테스트 완료")
        else:
            print("\n주관식 테스트 실패")
            sys.exit(1)
    else:
        print(f"\n선택된 테스트: {test_size}문항")
        print("AI 추론 시스템을 테스트합니다...")

        success = run_test(test_size, verbose=True)

        if success:
            print(f"\n테스트 완료 - 결과 파일: {DEFAULT_FILES['test_output_file']}")
        else:
            print("\n테스트 실패")
            sys.exit(1)


if __name__ == "__main__":
    main()