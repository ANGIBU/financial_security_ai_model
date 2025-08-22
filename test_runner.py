# test_runner.py

"""
테스트 실행기 - 주관식 답변 생성 특화 테스트
- 주관식 답변 품질 집중 테스트
- 템플릿 활용 효과성 검증
- 의도 분석 정확도 측정
- 자연스러운 한국어 생성 확인
- 품질 향상 기능 테스트
"""

import os
import sys
from pathlib import Path
import time

# 현재 디렉토리 설정
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# 설정 파일 import
from config import FILE_VALIDATION, DEFAULT_FILES, print_config_summary, relax_quality_standards
from inference import FinancialAIInference


def run_enhanced_subjective_test(test_size: int = None, verbose: bool = True):
    """주관식 답변 생성 특화 테스트 실행"""

    # 기본 테스트 크기 설정
    if test_size is None:
        test_size = 20  # 주관식 특화 테스트는 더 적은 수로

    print(f"\n=== 주관식 답변 생성 특화 테스트 시작 ===")
    print(f"테스트 크기: {test_size}개 문항")
    print("주요 검증 항목:")
    print("- 템플릿 활용 효과성")
    print("- 의도 분석 정확도")
    print("- 자연스러운 한국어 생성")
    print("- 반복 패턴 방지")
    print("- 답변 품질 일관성")

    # 파일 존재 확인
    test_file = DEFAULT_FILES["test_file"]
    submission_file = DEFAULT_FILES["submission_file"]

    for file_path in [test_file, submission_file]:
        if not os.path.exists(file_path):
            print(f"오류: {file_path} 파일이 없습니다")
            return False

    engine = None
    try:
        # 설정 요약 출력
        if verbose:
            print_config_summary()

        # AI 엔진 초기화
        print("\n주관식 특화 AI 엔진 초기화 중...")
        engine = FinancialAIInference(verbose=verbose)

        # 테스트 데이터 준비
        import pandas as pd

        test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["encoding"])
        submission_df = pd.read_csv(
            submission_file, encoding=FILE_VALIDATION["encoding"]
        )

        print(f"전체 데이터: {len(test_df)}개 문항")

        # 주관식 문항 우선 필터링
        subjective_questions = []
        print("\n주관식 문항 필터링 중...")
        
        for idx, row in test_df.iterrows():
            question = row["Question"]
            question_type, _ = engine.data_processor.extract_choice_range(question)
            
            if question_type == "subjective":
                subjective_questions.append(idx)
            
            if len(subjective_questions) >= test_size:
                break
        
        if len(subjective_questions) == 0:
            print("주관식 문항을 찾을 수 없습니다. 처음 문항들을 주관식으로 처리합니다.")
            subjective_questions = list(range(min(test_size, len(test_df))))
        
        print(f"주관식 문항 {len(subjective_questions)}개 선별 완료")

        # 주관식 문항만 테스트
        subjective_test_df = test_df.iloc[subjective_questions].copy()
        subjective_submission_df = submission_df.iloc[subjective_questions].copy()

        output_file = "./enhanced_subjective_test_result.csv"
        
        # 테스트 실행 시간 측정
        start_time = time.time()
        results = engine.execute_inference_with_data(
            subjective_test_df, subjective_submission_df, output_file
        )
        end_time = time.time()

        # 결과 분석 및 출력
        print_enhanced_subjective_results(
            results, output_file, len(subjective_questions), 
            subjective_test_df["ID"].tolist(), end_time - start_time
        )

        # 추가 품질 분석
        analyze_subjective_quality(output_file, subjective_test_df, results)

        return True

    except Exception as e:
        print(f"주관식 특화 테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if engine:
            engine.cleanup()


def run_template_effectiveness_test(test_size: int = 10):
    """템플릿 활용 효과성 테스트"""
    
    print(f"\n=== 템플릿 활용 효과성 테스트 ===")
    print("템플릿 기반 답변 vs 일반 답변 비교 테스트")

    engine = None
    try:
        # AI 엔진 초기화
        engine = FinancialAIInference(verbose=True)

        # 테스트 데이터 준비
        import pandas as pd
        test_df = pd.read_csv(DEFAULT_FILES["test_file"])
        
        # 주관식 문항 선별
        subjective_indices = []
        for idx, row in test_df.iterrows():
            question = row["Question"]
            question_type, _ = engine.data_processor.extract_choice_range(question)
            if question_type == "subjective":
                subjective_indices.append(idx)
            if len(subjective_indices) >= test_size:
                break

        if not subjective_indices:
            subjective_indices = list(range(min(test_size, len(test_df))))

        print(f"템플릿 효과성 테스트: {len(subjective_indices)}개 문항")

        template_results = []
        for idx in subjective_indices:
            row = test_df.iloc[idx]
            question = row["Question"]
            question_id = row["ID"]
            
            print(f"\n테스트 문항: {question_id}")
            print(f"질문: {question[:100]}...")
            
            # 템플릿 기반 답변 생성
            answer = engine.process_single_question(question, question_id)
            
            # 답변 품질 분석
            korean_ratio = engine.data_processor.calculate_korean_ratio(answer)
            has_repetition = engine.model_handler.detect_critical_repetitive_patterns(answer)
            
            template_results.append({
                "question_id": question_id,
                "answer_length": len(answer),
                "korean_ratio": korean_ratio,
                "has_repetition": has_repetition,
                "answer_preview": answer[:150]
            })
            
            print(f"답변 길이: {len(answer)}")
            print(f"한국어 비율: {korean_ratio:.2%}")
            print(f"반복 패턴: {'있음' if has_repetition else '없음'}")
            print(f"답변: {answer[:100]}...")

        # 결과 요약
        print(f"\n=== 템플릿 효과성 테스트 결과 요약 ===")
        avg_length = sum(r["answer_length"] for r in template_results) / len(template_results)
        avg_korean_ratio = sum(r["korean_ratio"] for r in template_results) / len(template_results)
        repetition_count = sum(1 for r in template_results if r["has_repetition"])
        
        print(f"평균 답변 길이: {avg_length:.1f}자")
        print(f"평균 한국어 비율: {avg_korean_ratio:.1%}")
        print(f"반복 패턴 발생: {repetition_count}/{len(template_results)}개")
        print(f"품질 성공률: {((len(template_results) - repetition_count) / len(template_results)):.1%}")

        return True

    except Exception as e:
        print(f"템플릿 효과성 테스트 오류: {e}")
        return False
    finally:
        if engine:
            engine.cleanup()


def run_intent_analysis_accuracy_test(test_size: int = 15):
    """의도 분석 정확도 테스트"""
    
    print(f"\n=== 의도 분석 정확도 테스트 ===")
    print("질문 의도 분석과 답변 일치성 검증")

    engine = None
    try:
        # AI 엔진 초기화
        engine = FinancialAIInference(verbose=True)

        # 테스트 데이터 준비
        import pandas as pd
        test_df = pd.read_csv(DEFAULT_FILES["test_file"])
        
        # 다양한 의도의 문항 선별
        intent_results = {}
        processed_count = 0
        
        for idx, row in test_df.iterrows():
            if processed_count >= test_size:
                break
                
            question = row["Question"]
            question_id = row["ID"]
            question_type, _ = engine.data_processor.extract_choice_range(question)
            
            if question_type == "subjective":
                # 의도 분석
                intent_analysis = engine.data_processor.analyze_question_intent(question)
                primary_intent = intent_analysis.get("primary_intent", "일반")
                confidence = intent_analysis.get("intent_confidence", 0)
                
                if primary_intent not in intent_results:
                    intent_results[primary_intent] = []
                
                # 답변 생성
                answer = engine.process_single_question(question, question_id)
                
                # 의도-답변 일치성 검증
                intent_match = engine.data_processor.validate_answer_intent_match(
                    answer, question, intent_analysis
                )
                
                intent_results[primary_intent].append({
                    "question_id": question_id,
                    "confidence": confidence,
                    "intent_match": intent_match,
                    "answer_length": len(answer)
                })
                
                print(f"문항 {question_id}: {primary_intent} (신뢰도: {confidence:.2f}, 일치: {intent_match})")
                processed_count += 1

        # 결과 분석
        print(f"\n=== 의도 분석 정확도 결과 ===")
        total_matches = 0
        total_questions = 0
        
        for intent, results in intent_results.items():
            match_count = sum(1 for r in results if r["intent_match"])
            avg_confidence = sum(r["confidence"] for r in results) / len(results)
            
            print(f"{intent}: {match_count}/{len(results)} 일치 "
                  f"(성공률: {match_count/len(results):.1%}, 평균 신뢰도: {avg_confidence:.2f})")
            
            total_matches += match_count
            total_questions += len(results)
        
        overall_accuracy = total_matches / total_questions if total_questions > 0 else 0
        print(f"\n전체 의도-답변 일치 정확도: {overall_accuracy:.1%}")

        return True

    except Exception as e:
        print(f"의도 분석 정확도 테스트 오류: {e}")
        return False
    finally:
        if engine:
            engine.cleanup()


def run_test(test_size: int = None, verbose: bool = True, relax_standards: bool = False):
    """기존 호환성을 위한 일반 테스트 실행"""

    # 기본 테스트 크기 설정
    if test_size is None:
        test_size = 50

    # 품질 기준 완화 옵션
    if relax_standards:
        print("품질 기준을 완화하여 실행합니다...")
        relax_quality_standards()

    # 파일 존재 확인
    test_file = DEFAULT_FILES["test_file"]
    submission_file = DEFAULT_FILES["submission_file"]

    for file_path in [test_file, submission_file]:
        if not os.path.exists(file_path):
            print(f"오류: {file_path} 파일이 없습니다")
            return False

    engine = None
    try:
        # 설정 요약 출력
        if verbose:
            print_config_summary()

        # AI 엔진 초기화
        print("\n시스템 초기화 중...")
        engine = FinancialAIInference(verbose=verbose)

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
        print_enhanced_results(results, output_file, test_size, verbose)

        return True

    except Exception as e:
        print(f"테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if engine:
            engine.cleanup()


def print_enhanced_subjective_results(
    results: dict, output_file: str, test_count: int, question_ids: list, execution_time: float
):
    """주관식 특화 테스트 결과 출력"""
    
    print(f"\n=== 주관식 답변 생성 특화 테스트 완료 ===")
    print(f"처리 문항: {test_count}개")
    print(f"처리 시간: {execution_time:.1f}초")
    print(f"평균 문항당 시간: {execution_time/test_count:.1f}초")
    print(f"결과 파일: {output_file}")
    
    # 처리된 문항 ID 출력
    print(f"\n=== 처리된 문항 ID ===")
    print(f"총 {len(question_ids)}개 문항: {', '.join(question_ids[:10])}{'...' if len(question_ids) > 10 else ''}")
    
    # 주관식 특화 통계
    if "debug_counters" in results:
        debug_info = results["debug_counters"]
        enhancement_info = results.get("enhancement_applied", {})
        
        print(f"\n=== 주관식 답변 생성 특화 통계 ===")
        print(f"총 주관식 문항: {debug_info.get('subjective_questions', 0)}")
        print(f"템플릿 융합 활용: {enhancement_info.get('template_fusion', 0)}")
        print(f"자연스러운 생성: {enhancement_info.get('natural_generation', 0)}")
        print(f"품질 향상 적용: {enhancement_info.get('quality_enhancement', 0)}")
        print(f"한국어 최적화: {enhancement_info.get('korean_optimization', 0)}")
        print(f"폴백 사용: {debug_info.get('fallback_used', 0)}")
        
        # 성능 지표 계산
        subjective_count = debug_info.get('subjective_questions', test_count)
        if subjective_count > 0:
            template_fusion_rate = enhancement_info.get('template_fusion', 0) / subjective_count
            quality_enhancement_rate = enhancement_info.get('quality_enhancement', 0) / subjective_count
            korean_optimization_rate = enhancement_info.get('korean_optimization', 0) / subjective_count
            
            print(f"\n=== 성능 지표 ===")
            print(f"템플릿 융합 활용률: {template_fusion_rate:.1%}")
            print(f"품질 향상 적용률: {quality_enhancement_rate:.1%}")
            print(f"한국어 최적화율: {korean_optimization_rate:.1%}")
            
            # 성능 평가
            if template_fusion_rate >= 0.8:
                print("✅ 템플릿 활용이 매우 효과적입니다")
            elif template_fusion_rate >= 0.6:
                print("✅ 템플릿 활용이 양호합니다")
            else:
                print("⚠️  템플릿 활용률을 개선할 필요가 있습니다")
            
            if quality_enhancement_rate >= 0.9:
                print("✅ 품질 향상 시스템이 우수합니다")
            elif quality_enhancement_rate >= 0.7:
                print("✅ 품질 향상 시스템이 양호합니다")
            else:
                print("⚠️  품질 향상 시스템을 개선할 필요가 있습니다")

    print("="*60)


def analyze_subjective_quality(output_file: str, test_df, results: dict):
    """주관식 답변 품질 분석"""
    
    print(f"\n=== 주관식 답변 품질 분석 ===")
    
    try:
        import pandas as pd
        result_df = pd.read_csv(output_file)
        
        quality_metrics = {
            "total_answers": len(result_df),
            "empty_answers": 0,
            "short_answers": 0,  # 30자 미만
            "optimal_answers": 0,  # 30-300자
            "long_answers": 0,  # 300자 초과
            "korean_dominant": 0,  # 한국어 비율 80% 이상
            "natural_sentences": 0,  # 자연스러운 문장 구조
        }
        
        for idx, row in result_df.iterrows():
            answer = str(row.get("Answer", ""))
            
            if not answer or answer.strip() == "":
                quality_metrics["empty_answers"] += 1
                continue
            
            length = len(answer)
            if length < 30:
                quality_metrics["short_answers"] += 1
            elif length <= 300:
                quality_metrics["optimal_answers"] += 1
            else:
                quality_metrics["long_answers"] += 1
            
            # 한국어 비율 계산
            korean_chars = len([c for c in answer if '\uAC00' <= c <= '\uD7A3'])
            total_chars = len([c for c in answer if c.isalpha()])
            korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
            
            if korean_ratio >= 0.8:
                quality_metrics["korean_dominant"] += 1
            
            # 자연스러운 문장 구조 확인
            if (answer.endswith((".", "다", "요", "함", "니다", "습니다")) and 
                "." in answer and 
                not any(problem in answer for problem in ["갈취", "묻고"])):
                quality_metrics["natural_sentences"] += 1
        
        # 품질 지표 출력
        total = quality_metrics["total_answers"]
        print(f"총 답변 수: {total}")
        print(f"빈 답변: {quality_metrics['empty_answers']} ({quality_metrics['empty_answers']/total:.1%})")
        print(f"짧은 답변 (30자 미만): {quality_metrics['short_answers']} ({quality_metrics['short_answers']/total:.1%})")
        print(f"적정 답변 (30-300자): {quality_metrics['optimal_answers']} ({quality_metrics['optimal_answers']/total:.1%})")
        print(f"긴 답변 (300자 초과): {quality_metrics['long_answers']} ({quality_metrics['long_answers']/total:.1%})")
        print(f"한국어 우수 (80% 이상): {quality_metrics['korean_dominant']} ({quality_metrics['korean_dominant']/total:.1%})")
        print(f"자연스러운 문장: {quality_metrics['natural_sentences']} ({quality_metrics['natural_sentences']/total:.1%})")
        
        # 전체 품질 점수 계산
        quality_score = (
            quality_metrics['optimal_answers'] * 3 +
            quality_metrics['korean_dominant'] * 2 + 
            quality_metrics['natural_sentences'] * 2 +
            quality_metrics['long_answers'] * 1 -
            quality_metrics['short_answers'] * 1 -
            quality_metrics['empty_answers'] * 3
        ) / (total * 8) * 100
        
        print(f"\n전체 품질 점수: {quality_score:.1f}/100")
        
        if quality_score >= 80:
            print("🌟 우수한 답변 품질입니다!")
        elif quality_score >= 60:
            print("✅ 양호한 답변 품질입니다.")
        elif quality_score >= 40:
            print("⚠️  답변 품질 개선이 필요합니다.")
        else:
            print("❌ 답변 품질이 낮습니다. 시스템 점검이 필요합니다.")
            
    except Exception as e:
        print(f"품질 분석 중 오류: {e}")


def print_enhanced_results(results: dict, output_file: str, test_count: int, verbose: bool = True):
    """향상된 결과 출력 (기존 호환성)"""
    
    total_time_minutes = results["total_time"] / 60
    print(f"\n=== 테스트 완료 ({test_count}개 문항) ===")
    print(f"처리 시간: {total_time_minutes:.2f}분")
    print(f"평균 문항당 시간: {results['total_time']/test_count:.2f}초")
    print(f"결과 파일: {output_file}")
    
    # 디버깅 통계 출력
    if "debug_counters" in results and verbose:
        debug_info = results["debug_counters"]
        print(f"\n=== 상세 처리 통계 ===")
        print(f"총 질문 수: {debug_info.get('total_questions', 0)}")
        print(f"주관식 질문: {debug_info.get('subjective_questions', 0)}")
        print(f"템플릿 활용: {debug_info.get('template_used', 0)}")
        print(f"폴백 사용: {debug_info.get('fallback_used', 0)}")
        print(f"품질 검증 통과: {debug_info.get('quality_passed', 0)}")
        
        # 향상된 통계 출력
        if "enhancement_applied" in results:
            enhancement_info = results["enhancement_applied"]
            print(f"\n=== 답변 생성 강화 통계 ===")
            print(f"템플릿 융합: {enhancement_info.get('template_fusion', 0)}")
            print(f"자연스러운 생성: {enhancement_info.get('natural_generation', 0)}")
            print(f"품질 향상: {enhancement_info.get('quality_enhancement', 0)}")
            print(f"한국어 최적화: {enhancement_info.get('korean_optimization', 0)}")
    
    print("="*60)


def select_test_type():
    """테스트 유형 선택"""
    print("\n=== AI 금융보안 테스트 시스템 (주관식 특화 버전) ===")
    print("실행할 테스트를 선택하세요:")
    print()
    print("1. 주관식 답변 생성 특화 테스트 (추천)")
    print("2. 템플릿 활용 효과성 테스트")
    print("3. 의도 분석 정확도 테스트")
    print("4. 기본 통합 테스트")
    print()

    while True:
        try:
            choice = input("선택 (1-4): ").strip()

            if choice == "1":
                return "subjective_enhanced"
            elif choice == "2":
                return "template_effectiveness"
            elif choice == "3":
                return "intent_accuracy"
            elif choice == "4":
                return "basic_test"
            else:
                print("잘못된 선택입니다. 1, 2, 3, 4 중 하나를 입력하세요.")

        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            sys.exit(0)
        except Exception:
            print("잘못된 입력입니다. 다시 시도하세요.")


def select_test_size(test_type: str):
    """테스트 크기 선택"""
    print(f"\n{test_type} 테스트 크기를 선택하세요:")
    
    if test_type == "주관식 특화":
        options = {
            "1": 5,
            "2": 10,
            "3": 20,
            "4": 30,
            "5": 50
        }
        print("1. 5문항 (빠른 확인)")
        print("2. 10문항 (기본 테스트)")
        print("3. 20문항 (상세 테스트)")
        print("4. 30문항 (종합 테스트)")
        print("5. 50문항 (전체 평가)")
    else:
        options = {
            "1": 5,
            "2": 10,
            "3": 15,
            "4": 25,
            "5": 50
        }
        print("1. 5문항 (빠른 테스트)")
        print("2. 10문항 (기본 테스트)")
        print("3. 15문항 (상세 테스트)")
        print("4. 25문항 (종합 테스트)")
        print("5. 50문항 (전체 테스트)")
    
    print()

    while True:
        try:
            choice = input("선택 (1-5): ").strip()

            if choice in options:
                return options[choice]
            else:
                print("잘못된 선택입니다. 1, 2, 3, 4, 5 중 하나를 입력하세요.")

        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            sys.exit(0)
        except Exception:
            print("잘못된 입력입니다. 다시 시도하세요.")


def main():
    """메인 함수"""
    
    # 테스트 유형 선택
    test_type = select_test_type()
    
    if test_type == "subjective_enhanced":
        test_size = select_test_size("주관식 특화")
        print(f"\n주관식 답변 생성 특화 테스트를 실행합니다... ({test_size}문항)")
        success = run_enhanced_subjective_test(test_size, verbose=True)
        
    elif test_type == "template_effectiveness":
        test_size = select_test_size("템플릿 효과성")
        print(f"\n템플릿 활용 효과성 테스트를 실행합니다... ({test_size}문항)")
        success = run_template_effectiveness_test(test_size)
        
    elif test_type == "intent_accuracy":
        test_size = select_test_size("의도 분석")
        print(f"\n의도 분석 정확도 테스트를 실행합니다... ({test_size}문항)")
        success = run_intent_analysis_accuracy_test(test_size)
        
    else:  # basic_test
        test_size = select_test_size("기본 통합")
        print(f"\n기본 통합 테스트를 실행합니다... ({test_size}문항)")
        success = run_test(test_size, verbose=True)
    
    if success:
        print(f"\n테스트 완료!")
    else:
        print(f"\n테스트 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()