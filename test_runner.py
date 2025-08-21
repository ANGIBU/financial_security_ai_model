# test_runner.py

"""
테스트 실행기
- 시스템 테스트 실행
- 의도 일치 성공률 표시
- 핵심 성능 지표 출력
- 주관식 전용 테스트 기능
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
    """기본 테스트 실행"""

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
        submission_df = pd.read_csv(submission_file, encoding=FILE_VALIDATION["encoding"])

        print(f"전체 데이터: {len(test_df)}개 문항")
        print(f"테스트 크기: {test_size}개 문항")

        # 지정된 크기로 제한
        if len(test_df) > test_size:
            test_df = test_df.head(test_size)
            temp_submission = submission_df.head(test_size).copy()

            output_file = DEFAULT_FILES["test_output_file"]
            results = engine.execute_inference_with_data(test_df, temp_submission, output_file)
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


def run_subjective_test(target_ids: list = None, verbose: bool = True):
    """주관식 전용 테스트 실행"""
    
    # 기본 주관식 문항 ID
    if target_ids is None:
        target_ids = ["test_004", "test_007"]

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
        submission_df = pd.read_csv(submission_file, encoding=FILE_VALIDATION["encoding"])

        # 주관식 문항 필터링
        subjective_test_df = test_df[test_df['ID'].isin(target_ids)].copy()
        subjective_submission_df = submission_df[submission_df['ID'].isin(target_ids)].copy()

        if len(subjective_test_df) == 0:
            print(f"오류: 지정된 주관식 문항을 찾을 수 없습니다: {target_ids}")
            return False

        print(f"주관식 테스트 문항: {len(subjective_test_df)}개")
        print(f"대상 문항 ID: {target_ids}")

        # 문항 내용 미리보기
        print("\n=== 테스트 문항 미리보기 ===")
        for idx, row in subjective_test_df.iterrows():
            question_id = row['ID']
            question_text = row['Question']
            print(f"\n[{question_id}]")
            print(f"질문: {question_text[:100]}{'...' if len(question_text) > 100 else ''}")

        # 주관식 테스트 실행
        output_file = "./subjective_test_result.csv"
        results = engine.execute_inference_with_data(subjective_test_df, subjective_submission_df, output_file)

        # 주관식 전용 결과 분석
        print_subjective_results(results, output_file, target_ids, subjective_test_df)

        return True

    except Exception as e:
        print(f"주관식 테스트 실행 오류: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        if engine:
            engine.cleanup()


def analyze_question_types(test_df):
    """문항 유형 자동 분석"""
    
    objective_patterns = [
        r'\b[1-5]\.',  # 1. 2. 3. 등의 패턴
        r'\([1-5]\)',  # (1) (2) (3) 등의 패턴
        r'①|②|③|④|⑤',  # 원 숫자 패턴
        r'다음\s*중',  # "다음 중" 표현
        r'선택하시오',  # "선택하시오" 표현
    ]
    
    objective_ids = []
    subjective_ids = []
    
    import re
    
    for idx, row in test_df.iterrows():
        question_id = row['ID']
        question_text = row['Question']
        
        # 객관식 패턴 검사
        is_objective = any(re.search(pattern, question_text) for pattern in objective_patterns)
        
        if is_objective:
            objective_ids.append(question_id)
        else:
            subjective_ids.append(question_id)
    
    return objective_ids, subjective_ids


def run_auto_subjective_test(verbose: bool = True):
    """자동 주관식 문항 탐지 및 테스트"""
    
    # 파일 존재 확인
    test_file = DEFAULT_FILES["test_file"]
    
    if not os.path.exists(test_file):
        print(f"오류: {test_file} 파일이 없습니다")
        return False

    try:
        # 테스트 데이터 로드
        import pandas as pd
        test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["encoding"])
        
        # 문항 유형 자동 분석
        print("\n문항 유형 자동 분석 중...")
        objective_ids, subjective_ids = analyze_question_types(test_df)
        
        print(f"객관식 문항: {len(objective_ids)}개")
        print(f"주관식 문항: {len(subjective_ids)}개")
        
        if len(subjective_ids) == 0:
            print("주관식 문항이 발견되지 않았습니다.")
            return False
        
        print(f"\n발견된 주관식 문항 ID: {subjective_ids[:10]}{'...' if len(subjective_ids) > 10 else ''}")
        
        # 주관식 테스트 실행
        return run_subjective_test(subjective_ids, verbose)
        
    except Exception as e:
        print(f"자동 주관식 테스트 오류: {e}")
        return False


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

    print(f"\n=== 객관식 성능 (전체의 {(mc_count/results['total_questions']*100):.0f}%) ===")
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

    print(f"\n=== 주관식 성능 (전체의 {(subj_count/results['total_questions']*100):.0f}%) ===")
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


def print_subjective_results(results: dict, output_file: str, target_ids: list, test_df):
    """주관식 전용 결과 분석"""
    
    total_time_minutes = results["total_time"] / 60
    total_questions = results["total_questions"]
    
    print(f"\n=== 주관식 테스트 완료 ===")
    print(f"처리 시간: {total_time_minutes:.1f}분")
    print(f"처리 문항: {total_questions}개")
    print(f"테스트 문항 ID: {', '.join(target_ids)}")

    # 주관식 성능 상세 분석
    subj_count = results.get("subj_count", 0)
    intent_success_rate = results.get("intent_match_success_rate", 0)
    korean_compliance = results.get("korean_compliance_rate", 0)
    avg_quality = results.get("avg_quality_score", 0)
    
    print(f"\n=== 주관식 성능 상세 분석 ===")
    print(f"주관식 문항 수: {subj_count}개")
    print(f"의도 일치 성공률: {intent_success_rate:.1f}%")
    print(f"한국어 준수율: {korean_compliance:.1f}%")
    print(f"평균 품질 점수: {avg_quality:.3f}/1.0")
    
    # 의도별 품질 분석
    intent_quality = results.get("intent_quality_by_type", {})
    if intent_quality:
        print(f"\n=== 의도별 품질 분석 ===")
        for intent_type, quality_score in intent_quality.items():
            print(f"{intent_type}: {quality_score:.3f}/1.0")
    
    # 템플릿 및 기관 질문 분석
    institution_count = results.get("institution_questions_count", 0)
    template_usage = results.get("template_usage_rate", 0)
    template_guided = results.get("template_guided_answer_rate", 0)
    
    print(f"\n=== 주관식 특화 기능 ===")
    print(f"기관 관련 질문: {institution_count}개")
    print(f"템플릿 활용률: {template_usage:.1f}%")
    print(f"템플릿 가이드 답변률: {template_guided:.1f}%")
    
    # 텍스트 품질 개선 분석
    text_recovery = results.get("text_recovery_rate", 0)
    grammar_fix = results.get("grammar_fix_rate", 0)
    korean_enhancement = results.get("korean_enhancement_count", 0)
    structure_improvement = results.get("answer_structure_improvement_rate", 0)
    
    print(f"\n=== 텍스트 품질 개선 ===")
    print(f"텍스트 복구율: {text_recovery:.1f}%")
    print(f"문법 수정률: {grammar_fix:.1f}%")
    print(f"한국어 강화 횟수: {korean_enhancement}회")
    print(f"답변 구조 개선률: {structure_improvement:.1f}%")
    
    # 개별 문항 분석
    print(f"\n=== 개별 문항 분석 ===")
    
    # 결과 파일에서 답변 읽기
    try:
        import pandas as pd
        result_df = pd.read_csv(output_file, encoding=FILE_VALIDATION["encoding"])
        
        for target_id in target_ids:
            # 원본 질문 찾기
            question_row = test_df[test_df['ID'] == target_id]
            if not question_row.empty:
                question_text = question_row.iloc[0]['Question']
                
                # 생성된 답변 찾기
                answer_row = result_df[result_df['ID'] == target_id]
                if not answer_row.empty:
                    answer_text = str(answer_row.iloc[0]['Answer'])
                    
                    print(f"\n[{target_id}]")
                    print(f"질문: {question_text[:150]}{'...' if len(question_text) > 150 else ''}")
                    print(f"답변: {answer_text[:200]}{'...' if len(answer_text) > 200 else ''}")
                    print(f"답변 길이: {len(answer_text)}자")
                    
                    # 한국어 비율 계산
                    korean_chars = sum(1 for char in answer_text if '가' <= char <= '힣')
                    total_chars = len(answer_text.replace(' ', ''))
                    korean_ratio = (korean_chars / max(total_chars, 1)) * 100
                    print(f"한국어 비율: {korean_ratio:.1f}%")
                    
    except Exception as e:
        print(f"개별 문항 분석 중 오류: {e}")
    
    # 성능 개선 제안
    print_subjective_improvement_suggestions(results)
    
    print(f"\n결과 파일 저장됨: {output_file}")


def print_subjective_improvement_suggestions(results: dict):
    """주관식 성능 개선 제안"""
    
    print(f"\n=== 주관식 성능 개선 제안 ===")
    
    intent_rate = results.get("intent_match_success_rate", 0)
    korean_compliance = results.get("korean_compliance_rate", 0)
    avg_quality = results.get("avg_quality_score", 0)
    template_usage = results.get("template_usage_rate", 0)
    
    suggestions = []
    
    if intent_rate < 80:
        suggestions.append(f"의도 일치율이 {intent_rate:.1f}%로 낮습니다. 질문 의도 분석 알고리즘을 개선하세요.")
    
    if korean_compliance < 90:
        suggestions.append(f"한국어 준수율이 {korean_compliance:.1f}%입니다. 한국어 텍스트 생성을 강화하세요.")
    
    if avg_quality < 0.8:
        suggestions.append(f"평균 품질 점수가 {avg_quality:.2f}로 낮습니다. 답변 구조와 내용을 개선하세요.")
    
    if template_usage < 50:
        suggestions.append(f"템플릿 활용률이 {template_usage:.1f}%로 낮습니다. 도메인별 템플릿을 확장하세요.")
    
    if not suggestions:
        suggestions.append("주관식 성능이 우수합니다. 현재 수준을 유지하세요.")
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")


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


def select_test_mode():
    """테스트 모드 선택"""
    print("\n테스트 모드를 선택하세요:")
    print("1. 전체 테스트 (객관식 + 주관식)")
    print("2. 주관식 전용 테스트 (test_004, test_007)")
    print("3. 자동 주관식 테스트 (주관식 문항 자동 탐지)")
    print("4. 미니 테스트 (8문항)")
    print("5. 기본 테스트 (50문항)")
    print("6. 정밀 테스트 (100문항)")
    print()

    while True:
        try:
            choice = input("선택 (1-6): ").strip()

            if choice == "1":
                test_size = select_test_size()
                return "full", test_size
            elif choice == "2":
                return "subjective", None
            elif choice == "3":
                return "auto_subjective", None
            elif choice == "4":
                return "full", TEST_CONFIG["test_sizes"]["mini"]
            elif choice == "5":
                return "full", TEST_CONFIG["test_sizes"]["basic"]
            elif choice == "6":
                return "full", TEST_CONFIG["test_sizes"]["detailed"]
            else:
                print("잘못된 선택입니다. 1-6 중 하나를 입력하세요.")

        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            sys.exit(0)
        except Exception:
            print("잘못된 입력입니다. 다시 시도하세요.")


def select_test_size():
    """테스트 문항 수 선택"""
    print("\n테스트할 문항 수를 선택하세요:")

    test_options = TEST_CONFIG["test_sizes"]
    print(f"1. {test_options['mini']}문항 (미니 테스트)")
    print(f"2. {test_options['basic']}문항 (기본 테스트)")
    print(f"3. {test_options['detailed']}문항 (정밀 테스트)")
    print()

    while True:
        try:
            choice = input("선택 (1-3): ").strip()

            if choice == "1":
                return test_options["mini"]
            elif choice == "2":
                return test_options["basic"]
            elif choice == "3":
                return test_options["detailed"]
            else:
                print("잘못된 선택입니다. 1, 2, 3 중 하나를 입력하세요.")

        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            sys.exit(0)
        except Exception:
            print("잘못된 입력입니다. 다시 시도하세요.")


def main():
    """메인 함수"""
    
    print("=== 금융보안 AI 추론 시스템 테스트 ===")
    
    # 테스트 모드 선택
    test_mode, test_size = select_test_mode()
    
    if test_mode == "full":
        print(f"\n선택된 테스트: 전체 테스트 ({test_size}문항)")
        print("AI 추론 시스템을 테스트합니다...")
        success = run_test(test_size, verbose=True)
        
        if success:
            print(f"\n테스트 완료 - 결과 파일: {DEFAULT_FILES['test_output_file']}")
        else:
            print("\n테스트 실패")
            sys.exit(1)
            
    elif test_mode == "subjective":
        print(f"\n선택된 테스트: 주관식 전용 테스트 (test_004, test_007)")
        print("주관식 전용 AI 추론 시스템을 테스트합니다...")
        success = run_subjective_test(["test_004", "test_007"], verbose=True)
        
        if success:
            print(f"\n주관식 테스트 완료 - 결과 파일: ./subjective_test_result.csv")
        else:
            print("\n주관식 테스트 실패")
            sys.exit(1)
            
    elif test_mode == "auto_subjective":
        print(f"\n선택된 테스트: 자동 주관식 테스트")
        print("주관식 문항을 자동으로 탐지하여 테스트합니다...")
        success = run_auto_subjective_test(verbose=True)
        
        if success:
            print(f"\n자동 주관식 테스트 완료 - 결과 파일: ./subjective_test_result.csv")
        else:
            print("\n자동 주관식 테스트 실패")
            sys.exit(1)


if __name__ == "__main__":
    main()