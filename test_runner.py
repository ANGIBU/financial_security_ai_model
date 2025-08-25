# test_runner.py

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

            # 상세 결과 출력
            print_results(results, output_file, test_size)
            
            # 답변 분석 수행
            if Path(output_file).exists():
                try:
                    print_answer_analysis(engine, test_df, output_file)
                    if test_size <= 10:  # 작은 테스트에서만 주관식 분석 수행
                        print_subjective_answer_analysis(engine, test_df, output_file)
                except Exception as e:
                    print(f"답변 분석 중 오류: {e}")

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

            found_ids = comprehensive_test_df["ID"].tolist()

            output_file = current_dir / "comprehensive_test_result.csv"
            results = engine.execute_inference_with_data(
                comprehensive_test_df, comprehensive_submission_df, output_file
            )

            # 종합 결과 분석
            print_comprehensive_results(results, output_file, len(comprehensive_test_df), found_ids)
            
            # 상세 분석 수행
            try:
                analyze_comprehensive_results(engine, comprehensive_test_df, output_file, results)
            except Exception as e:
                print(f"종합 분석 중 오류: {e}")

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

            # 타입별 결과 분석
            if question_type == "주관식":
                print_subjective_results(results, output_file, len(type_indices), type_questions)
                # 주관식 상세 분석
                try:
                    print_subjective_answer_analysis(engine, type_test_df, output_file)
                except Exception as e:
                    print(f"주관식 분석 중 오류: {e}")
            else:
                print_multiple_choice_results(results, output_file, len(type_indices), type_questions)
                # 객관식 상세 분석
                try:
                    analyze_multiple_choice_results(engine, type_test_df, output_file, results)
                except Exception as e:
                    print(f"객관식 분석 중 오류: {e}")

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


def print_answer_analysis(engine, test_df, output_file):
    """답변 분석 (엔진 재사용)"""
    try:
        import pandas as pd
        
        if not Path(output_file).exists():
            print("결과 파일이 존재하지 않습니다.")
            return
            
        result_df = pd.read_csv(output_file, encoding=FILE_VALIDATION["encoding"])
        
        print("\n=== 답변 분석 ===")
        
        subjective_count = 0
        objective_count = 0
        fallback_count = 0
        template_count = 0
        error_count = 0
        
        domain_stats = {}
        method_stats = {}
        
        for idx, row in result_df.iterrows():
            question_id = row["ID"]
            answer = str(row["Answer"]) if pd.notna(row["Answer"]) else ""
            
            # 원본 질문 찾기
            question_row = test_df[test_df["ID"] == question_id]
            if len(question_row) == 0:
                continue
                
            question = question_row.iloc[0]["Question"]
            
            try:
                question_type, max_choice = engine.data_processor.extract_choice_range(question)
                domain = engine.data_processor.extract_domain(question)
                
                # 도메인 통계
                if domain not in domain_stats:
                    domain_stats[domain] = {"count": 0, "subjective": 0, "objective": 0}
                domain_stats[domain]["count"] += 1
                
                if question_type == "subjective":
                    subjective_count += 1
                    domain_stats[domain]["subjective"] += 1
                    
                    # 답변 유형 분석
                    if not answer or len(answer.strip()) < 10:
                        error_count += 1
                    elif any(pattern in answer for pattern in [
                        "관련 법령과 규정에 따라 체계적인 관리",
                        "체계적이고 전문적인 관리",
                        "지속적으로 운영해야 합니다"
                    ]):
                        fallback_count += 1
                    elif any(indicator in answer for indicator in [
                        "트로이 목마 기반 원격제어 악성코드",
                        "전자금융분쟁조정위원회",
                        "개인정보보호위원회",
                        "정보보안관리체계"
                    ]):
                        template_count += 1
                        
                else:
                    objective_count += 1
                    domain_stats[domain]["objective"] += 1
                    
                    # 객관식 답변 검증
                    if not (answer.isdigit() and 1 <= int(answer) <= max_choice):
                        error_count += 1
                        
            except Exception as e:
                print(f"답변 분석 오류 ({question_id}): {e}")
                error_count += 1
                continue
                
        # 결과 출력
        print(f"주관식: {subjective_count}개")
        print(f"객관식: {objective_count}개")
        print(f"오류 답변: {error_count}개")
        
        if subjective_count > 0:
            fallback_rate = fallback_count / subjective_count * 100
            template_rate = template_count / subjective_count * 100
            success_rate = (subjective_count - fallback_count - error_count) / subjective_count * 100
            
            print(f"템플릿 기반 답변: {template_count}개 ({template_rate:.1f}%)")
            print(f"폴백 답변: {fallback_count}개 ({fallback_rate:.1f}%)")
            print(f"주관식 성공률: {success_rate:.1f}%")
        
        # 도메인별 분석
        print("\n=== 도메인별 분석 ===")
        for domain, stats in domain_stats.items():
            print(f"{domain}: 총 {stats['count']}개 (주관식 {stats['subjective']}, 객관식 {stats['objective']})")
        
        print("답변 분석 완료!")
        
    except Exception as e:
        print(f"답변 분석 중 오류: {e}")


def print_subjective_answer_analysis(engine, test_df, output_file):
    """주관식 답변 분석"""
    try:
        import pandas as pd
        
        if not Path(output_file).exists():
            print("결과 파일이 존재하지 않습니다.")
            return
            
        result_df = pd.read_csv(output_file, encoding=FILE_VALIDATION["encoding"])
        
        print("\n=== 주관식 답변 상세 분석 ===")
        
        fallback_answers = []
        good_answers = []
        template_answers = []
        error_answers = []
        
        for idx, row in result_df.iterrows():
            question_id = row["ID"]
            answer = str(row["Answer"]) if pd.notna(row["Answer"]) else ""
            
            # 원본 질문 찾기
            question_row = test_df[test_df["ID"] == question_id]
            if len(question_row) == 0:
                continue
                
            question = question_row.iloc[0]["Question"]
            
            try:
                question_type, max_choice = engine.data_processor.extract_choice_range(question)
                
                if question_type != "subjective":
                    continue
                
                # 답변 유형 분류
                if not answer or len(answer.strip()) < 10:
                    error_answers.append(question_id)
                elif any(pattern in answer for pattern in [
                    "관련 법령과 규정에 따라 체계적인 관리가 필요합니다",
                    "관련 법령과 규정에 따라 체계적인 관리를",
                    "체계적인 관리가 필요합니다",
                    "체계적이고 전문적인 관리",
                    "지속적으로 운영해야 합니다"
                ]):
                    fallback_answers.append(question_id)
                elif any(indicator in answer for indicator in [
                    "트로이 목마 기반 원격제어 악성코드(RAT)는",
                    "전자금융분쟁조정위원회에서",
                    "개인정보보호위원회에서 개인정보 보호에 관한",
                    "정보보안관리체계를 구축하여"
                ]):
                    template_answers.append(question_id)
                else:
                    good_answers.append(question_id)
                    
            except Exception as e:
                print(f"주관식 분석 오류 ({question_id}): {e}")
                error_answers.append(question_id)
                continue
                
        print(f"템플릿 기반 답변: {len(template_answers)}개")
        print(f"정상 LLM 답변: {len(good_answers)}개")
        print(f"폴백 답변: {len(fallback_answers)}개")
        print(f"오류 답변: {len(error_answers)}개")
        
        if fallback_answers:
            print(f"폴백 답변 ID (처음 5개): {', '.join(fallback_answers[:5])}")
            
        if error_answers:
            print(f"오류 답변 ID (처음 5개): {', '.join(error_answers[:5])}")
            
        total_answers = len(template_answers) + len(good_answers) + len(fallback_answers) + len(error_answers)
        if total_answers > 0:
            success_rate = (len(template_answers) + len(good_answers)) / total_answers * 100
            template_rate = len(template_answers) / total_answers * 100
            error_rate = len(error_answers) / total_answers * 100
            print(f"주관식 전체 성공률: {success_rate:.1f}%")
            print(f"템플릿 활용률: {template_rate:.1f}%")
            print(f"오류율: {error_rate:.1f}%")
        
        print("상세 분석 완료!")
        
    except Exception as e:
        print(f"주관식 답변 분석 중 오류: {e}")


def analyze_multiple_choice_results(engine, test_df, output_file, results):
    """객관식 결과 분석"""
    try:
        import pandas as pd
        
        if not Path(output_file).exists():
            print("결과 파일이 존재하지 않습니다.")
            return
            
        result_df = pd.read_csv(output_file, encoding=FILE_VALIDATION["encoding"])
        
        print("\n=== 객관식 상세 분석 ===")
        
        valid_answers = 0
        invalid_answers = 0
        pattern_matches = 0
        domain_distribution = {}
        
        for idx, row in result_df.iterrows():
            question_id = row["ID"]
            answer = str(row["Answer"]) if pd.notna(row["Answer"]) else ""
            
            question_row = test_df[test_df["ID"] == question_id]
            if len(question_row) == 0:
                continue
                
            question = question_row.iloc[0]["Question"]
            
            try:
                question_type, max_choice = engine.data_processor.extract_choice_range(question)
                domain = engine.data_processor.extract_domain(question)
                
                if question_type != "multiple_choice":
                    continue
                
                # 도메인 분포
                if domain not in domain_distribution:
                    domain_distribution[domain] = {"count": 0, "valid": 0}
                domain_distribution[domain]["count"] += 1
                
                # 답변 유효성 검사
                if answer.isdigit() and 1 <= int(answer) <= max_choice:
                    valid_answers += 1
                    domain_distribution[domain]["valid"] += 1
                    
                    # 특별 패턴 매칭 확인
                    if engine._is_special_mc_pattern(question):
                        pattern_matches += 1
                else:
                    invalid_answers += 1
                    
            except Exception as e:
                print(f"객관식 분석 오류 ({question_id}): {e}")
                invalid_answers += 1
                continue
        
        total_mc = valid_answers + invalid_answers
        if total_mc > 0:
            print(f"유효한 답변: {valid_answers}개 ({valid_answers/total_mc*100:.1f}%)")
            print(f"유효하지 않은 답변: {invalid_answers}개 ({invalid_answers/total_mc*100:.1f}%)")
            print(f"특별 패턴 처리: {pattern_matches}개")
            
        print("\n=== 도메인별 객관식 성공률 ===")
        for domain, stats in domain_distribution.items():
            success_rate = (stats["valid"] / stats["count"]) * 100 if stats["count"] > 0 else 0
            print(f"{domain}: {stats['valid']}/{stats['count']} ({success_rate:.1f}%)")
            
        print("객관식 분석 완료!")
        
    except Exception as e:
        print(f"객관식 분석 중 오류: {e}")


def analyze_comprehensive_results(engine, test_df, output_file, results):
    """종합 결과 분석"""
    try:
        print("\n=== 종합 분석 ===")
        
        # 성능 지표 출력
        perf_metrics = results.get("performance_metrics", {})
        if perf_metrics:
            print("=== 성능 지표 ===")
            print(f"최소 처리 시간: {perf_metrics.get('min_processing_time', 0)}초")
            print(f"최대 처리 시간: {perf_metrics.get('max_processing_time', 0)}초")
            print(f"중간값: {perf_metrics.get('median_processing_time', 0)}초")
            print(f"효율 점수: {perf_metrics.get('efficiency_score', 0)}점")
        
        # 품질 지표 출력
        quality_metrics = results.get("quality_metrics", {})
        if quality_metrics:
            print("\n=== 품질 지표 ===")
            print(f"전체 성공률: {quality_metrics.get('overall_success_rate', 0)}%")
            print(f"오류율: {quality_metrics.get('error_rate', 0)}%")
            print(f"템플릿 활용: {quality_metrics.get('template_utilization', 0)}개")
            print(f"도메인 커버리지: {quality_metrics.get('domain_coverage', 0)}개")
        
        # 기본 답변 분석도 수행
        print_answer_analysis(engine, test_df, output_file)
        
        print("종합 분석 완료!")
        
    except Exception as e:
        print(f"종합 분석 중 오류: {e}")


def print_comprehensive_results(results: dict, output_file: Path, test_count: int, question_ids: list):
    """종합 테스트 결과 출력"""
    try:
        print(f"\n=== 종합 테스트 완료 ({test_count}개 문항) ===")
        print(f"처리 시간: {results.get('total_time', 0):.1f}초")
        print(f"평균 처리 시간: {results.get('avg_processing_time', 0):.2f}초/문항")
        print(f"성공률: {results.get('success_rate', 0):.1f}%")
        print(f"결과 파일: {output_file}")
        
        # 도메인 분포 출력
        domain_dist = results.get("domain_distribution", {})
        if domain_dist:
            print("\n=== 도메인 분포 ===")
            for domain, count in domain_dist.items():
                print(f"{domain}: {count}개")
        
        # 학습 데이터 정보
        learning_data = results.get("learning_data", {})
        if learning_data:
            print(f"\n=== 학습 데이터 ===")
            print(f"성공 답변 누적: {learning_data.get('successful_answers', 0)}개")
            print(f"실패 답변 누적: {learning_data.get('failed_answers', 0)}개")
        
        print("종합 테스트 완료")
    except Exception as e:
        print(f"종합 결과 출력 중 오류: {e}")
        print(f"처리 시간: {results.get('total_time', 0):.1f}초")
        print(f"결과 파일: {output_file}")
        print("종합 테스트 완료")


def print_multiple_choice_results(results: dict, output_file: Path, test_count: int, question_ids: list):
    """객관식 테스트 결과 출력"""
    try:
        print(f"\n=== 객관식 테스트 완료 ({test_count}개 문항) ===")
        print(f"처리 시간: {results.get('total_time', 0):.1f}초")
        print(f"평균 처리 시간: {results.get('avg_processing_time', 0):.2f}초/문항")
        print(f"성공률: {results.get('success_rate', 0):.1f}%")
        print(f"결과 파일: {output_file}")
        
        # 방법별 분포
        method_dist = results.get("method_distribution", {})
        if method_dist:
            print("\n=== 처리 방법 분포 ===")
            for method, count in method_dist.items():
                print(f"{method}: {count}개")
        
        print("객관식 테스트 완료")
    except Exception as e:
        print(f"객관식 결과 출력 중 오류: {e}")
        print(f"처리 시간: {results.get('total_time', 0):.1f}초")
        print(f"결과 파일: {output_file}")
        print("객관식 테스트 완료")


def print_subjective_results(results: dict, output_file: Path, test_count: int, question_ids: list):
    """주관식 테스트 결과 출력"""
    try:
        print(f"\n=== 주관식 테스트 완료 ({test_count}개 문항) ===")
        print(f"처리 시간: {results.get('total_time', 0):.1f}초")
        print(f"평균 처리 시간: {results.get('avg_processing_time', 0):.2f}초/문항")
        print(f"성공률: {results.get('success_rate', 0):.1f}%")
        print(f"결과 파일: {output_file}")
        
        # 도메인 분포
        domain_dist = results.get("domain_distribution", {})
        if domain_dist:
            print("\n=== 도메인 분포 ===")
            for domain, count in domain_dist.items():
                print(f"{domain}: {count}개")
        
        print("주관식 테스트 완료")
    except Exception as e:
        print(f"주관식 결과 출력 중 오류: {e}")
        print(f"처리 시간: {results.get('total_time', 0):.1f}초")
        print(f"결과 파일: {output_file}")
        print("주관식 테스트 완료")


def print_results(results: dict, output_file: str, test_size: int):
    """기본 결과 출력"""
    try:
        total_time_minutes = results.get("total_time", 0) / 60
        print(f"\n=== 테스트 완료 ({test_size}개 문항) ===")
        print(f"처리 시간: {total_time_minutes:.1f}분")
        print(f"결과 파일: {output_file}")
        
        # 성능 정보
        if 'avg_processing_time' in results:
            print(f"평균 문항 처리 시간: {results['avg_processing_time']:.2f}초")
        
        if 'success_rate' in results:
            print(f"성공률: {results['success_rate']:.1f}%")
            print(f"성공: {results.get('successful_processing', 0)}개")
            print(f"실패: {results.get('failed_processing', 0)}개")
        
        # 학습 데이터 정보
        learning_data = results.get('learning_data', {})
        if learning_data:
            print(f"pkl 학습 데이터 저장:")
            print(f"  - 성공 답변: {learning_data.get('successful_answers', 0)}개")
            print(f"  - 실패 답변: {learning_data.get('failed_answers', 0)}개")
            print(f"  - 질문 패턴: {learning_data.get('question_patterns', 0)}개")
    except Exception as e:
        print(f"결과 출력 중 오류: {e}")
        print(f"처리 시간: {results.get('total_time', 0) / 60:.1f}분")
        print(f"결과 파일: {output_file}")


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