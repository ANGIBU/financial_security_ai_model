# test_runner.py

import sys
import time
from pathlib import Path

current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from config import FILE_VALIDATION, DEFAULT_FILES, ensure_directories, get_domain_weight
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
            engine = FinancialAIInference(verbose=False)

            import pandas as pd

            try:
                test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["encoding"])
                submission_df = pd.read_csv(submission_file, encoding=FILE_VALIDATION["encoding"])
            except UnicodeDecodeError:
                print(f"인코딩 오류 발생, UTF-8으로 재시도...")
                test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["backup_encoding"])
                submission_df = pd.read_csv(submission_file, encoding=FILE_VALIDATION["backup_encoding"])
            except Exception as e:
                print(f"데이터 로드 실패: {e}")
                return False

            print(f"데이터 로드 완료: {len(test_df)}개 문항")

            if len(test_df) > test_size:
                # 샘플링 전략 적용
                sampled_indices = smart_sampling(test_df, test_size, engine)
                test_df = test_df.iloc[sampled_indices]
                temp_submission = submission_df.iloc[sampled_indices].copy()

                output_file = DEFAULT_FILES["test_output_file"]
                results = engine.execute_inference_with_data(
                    test_df, temp_submission, output_file
                )
            else:
                output_file = DEFAULT_FILES["test_output_file"]
                results = engine.execute_inference(test_file, submission_file, output_file)

            print_results(results, output_file, len(test_df))

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


def smart_sampling(test_df, test_size: int, engine) -> list:
    """스마트 샘플링 전략"""
    try:
        domain_distribution = {}
        question_types = {"objective": [], "subjective": []}
        
        # 질문 분석
        for idx, row in test_df.iterrows():
            question = row["Question"]
            
            try:
                question_type, max_choice = engine.data_processor.extract_choice_range(question)
                domain = engine.data_processor.extract_domain(question)
                
                if domain not in domain_distribution:
                    domain_distribution[domain] = []
                domain_distribution[domain].append(idx)
                
                if question_type == "multiple_choice":
                    question_types["objective"].append(idx)
                else:
                    question_types["subjective"].append(idx)
                    
            except Exception as e:
                print(f"질문 분석 오류 ({idx}): {e}")
                continue
        
        # 도메인별 비례 샘플링
        selected_indices = []
        total_questions = len(test_df)
        
        for domain, indices in domain_distribution.items():
            domain_weight = get_domain_weight(domain)
            priority_boost = domain_weight.get("priority_boost", 1.0)
            
            # 도메인 비율 계산 (가중치 적용)
            domain_ratio = (len(indices) / total_questions) * priority_boost
            domain_sample_size = max(1, int(test_size * domain_ratio))
            domain_sample_size = min(domain_sample_size, len(indices))
            
            # 랜덤 샘플링
            import random
            random.seed(42)  # 재현 가능한 결과
            sampled = random.sample(indices, domain_sample_size)
            selected_indices.extend(sampled)
        
        # 부족한 경우 추가 샘플링
        if len(selected_indices) < test_size:
            remaining_indices = [idx for idx in test_df.index if idx not in selected_indices]
            additional_needed = test_size - len(selected_indices)
            additional_needed = min(additional_needed, len(remaining_indices))
            
            if additional_needed > 0:
                import random
                additional_samples = random.sample(remaining_indices, additional_needed)
                selected_indices.extend(additional_samples)
        
        # 크기 초과 시 잘라내기
        selected_indices = selected_indices[:test_size]
        
        return selected_indices
    
    except Exception as e:
        print(f"스마트 샘플링 오류: {e}")
        # 폴백: 단순 순차 샘플링
        return list(range(min(test_size, len(test_df))))


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
            engine = FinancialAIInference(verbose=False)

            import pandas as pd

            try:
                test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["encoding"])
                submission_df = pd.read_csv(submission_file, encoding=FILE_VALIDATION["encoding"])
            except UnicodeDecodeError:
                print(f"인코딩 오류 발생, UTF-8으로 재시도...")
                test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["backup_encoding"])
                submission_df = pd.read_csv(submission_file, encoding=FILE_VALIDATION["backup_encoding"])
            except Exception as e:
                print(f"데이터 로드 실패: {e}")
                return False

            print(f"데이터 로드 완료: {len(test_df)}개 문항")

            if len(test_df) > test_size:
                # 종합 테스트를 위한 균형잡힌 샘플링
                comprehensive_indices = balanced_sampling(test_df, test_size, engine)
                comprehensive_test_df = test_df.iloc[comprehensive_indices].copy()
                comprehensive_submission_df = submission_df.iloc[comprehensive_indices].copy()
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


def balanced_sampling(test_df, test_size: int, engine) -> list:
    """균형잡힌 샘플링"""
    try:
        domain_question_map = {}
        type_question_map = {"objective": [], "subjective": []}
        
        # 질문 분류
        for idx, row in test_df.iterrows():
            question = row["Question"]
            
            try:
                question_type, max_choice = engine.data_processor.extract_choice_range(question)
                domain = engine.data_processor.extract_domain(question)
                
                if domain not in domain_question_map:
                    domain_question_map[domain] = {"objective": [], "subjective": []}
                
                if question_type == "multiple_choice":
                    domain_question_map[domain]["objective"].append(idx)
                    type_question_map["objective"].append(idx)
                else:
                    domain_question_map[domain]["subjective"].append(idx)
                    type_question_map["subjective"].append(idx)
                    
            except Exception as e:
                print(f"질문 분류 오류 ({idx}): {e}")
                continue
        
        # 균형잡힌 선택
        selected_indices = []
        domains = list(domain_question_map.keys())
        questions_per_domain = max(1, test_size // len(domains))
        
        import random
        random.seed(42)
        
        for domain in domains:
            domain_data = domain_question_map[domain]
            total_domain_questions = len(domain_data["objective"]) + len(domain_data["subjective"])
            
            if total_domain_questions == 0:
                continue
                
            # 도메인 내에서 객관식/주관식 균형
            domain_sample_size = min(questions_per_domain, total_domain_questions)
            obj_ratio = len(domain_data["objective"]) / total_domain_questions
            
            obj_count = max(1, int(domain_sample_size * obj_ratio))
            subj_count = domain_sample_size - obj_count
            
            # 객관식 샘플링
            if len(domain_data["objective"]) >= obj_count:
                obj_samples = random.sample(domain_data["objective"], obj_count)
                selected_indices.extend(obj_samples)
            else:
                selected_indices.extend(domain_data["objective"])
                
            # 주관식 샘플링
            if len(domain_data["subjective"]) >= subj_count:
                subj_samples = random.sample(domain_data["subjective"], subj_count)
                selected_indices.extend(subj_samples)
            else:
                selected_indices.extend(domain_data["subjective"])
        
        # 부족한 경우 보완
        if len(selected_indices) < test_size:
            remaining_indices = [idx for idx in test_df.index if idx not in selected_indices]
            additional_needed = test_size - len(selected_indices)
            additional_needed = min(additional_needed, len(remaining_indices))
            
            if additional_needed > 0:
                additional_samples = random.sample(remaining_indices, additional_needed)
                selected_indices.extend(additional_samples)
        
        selected_indices = selected_indices[:test_size]
        
        return selected_indices
    
    except Exception as e:
        print(f"균형 샘플링 오류: {e}")
        return list(range(min(test_size, len(test_df))))


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
            engine = FinancialAIInference(verbose=False)

            import pandas as pd

            try:
                test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["encoding"])
                submission_df = pd.read_csv(submission_file, encoding=FILE_VALIDATION["encoding"])
            except UnicodeDecodeError:
                print(f"인코딩 오류 발생, UTF-8으로 재시도...")
                test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["backup_encoding"])
                submission_df = pd.read_csv(submission_file, encoding=FILE_VALIDATION["backup_encoding"])
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
                fallback_size = test_size if test_size != "전체" else 50
                print(f"{question_type} 문항을 찾을 수 없어 처음 {fallback_size}개 문항으로 대체합니다.")
                type_indices = list(range(min(fallback_size, len(test_df))))
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
            engine = FinancialAIInference(verbose=False)

            import pandas as pd

            try:
                test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["encoding"])
                submission_df = pd.read_csv(submission_file, encoding=FILE_VALIDATION["encoding"])
            except UnicodeDecodeError:
                print(f"인코딩 오류 발생, UTF-8으로 재시도...")
                test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION["backup_encoding"])
                submission_df = pd.read_csv(submission_file, encoding=FILE_VALIDATION["backup_encoding"])
            except Exception as e:
                print(f"데이터 로드 실패: {e}")
                return False

            print(f"데이터 로드 완료: {len(test_df)}개 문항")

            domain_indices = []
            domain_questions = []
            question_type_counts = {"객관식": 0, "주관식": 0}
            difficulty_distribution = {"초급": 0, "중급": 0, "고급": 0}

            for idx, row in test_df.iterrows():
                question = row["Question"]
                question_id = row["ID"]

                try:
                    detected_domain = engine.data_processor.extract_domain(question)
                    
                    if detected_domain == domain_name:
                        detected_type, max_choice = engine.data_processor.extract_choice_range(question)
                        difficulty = engine.data_processor.analyze_question_difficulty(question)
                        
                        domain_indices.append(idx)
                        domain_questions.append(question_id)
                        
                        if detected_type == "multiple_choice":
                            question_type_counts["객관식"] += 1
                        else:
                            question_type_counts["주관식"] += 1
                        
                        # 난이도 분포
                        difficulty_distribution[difficulty] += 1

                    if len(domain_indices) >= test_size:
                        break
                        
                except Exception as e:
                    print(f"도메인 분석 오류 ({question_id}): {e}")
                    continue

            if len(domain_indices) == 0:
                print(f"{domain_name} 도메인 문항을 찾을 수 없습니다.")
                return False

            if len(domain_indices) > test_size:
                # 우선순위 기반 선택
                domain_weight = get_domain_weight(domain_name)
                if domain_weight.get("priority_boost", 1.0) > 1.2:
                    # 고우선순위 도메인은 다양한 샘플 선택
                    import random
                    random.seed(42)
                    domain_indices = random.sample(domain_indices, test_size)
                else:
                    domain_indices = domain_indices[:test_size]
                domain_questions = domain_questions[:test_size]

            print(f"\n{domain_name} 도메인 문항 분석:")
            print(f"  객관식: {question_type_counts['객관식']}개")
            print(f"  주관식: {question_type_counts['주관식']}개")
            print(f"  난이도 분포 - 초급: {difficulty_distribution['초급']}, 중급: {difficulty_distribution['중급']}, 고급: {difficulty_distribution['고급']}")

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
            success_rate = results.get('success_rate', 0)
            print(f"성공률: {success_rate:.1f}%")
            
            if success_rate >= 70:
                print("목표 성공률 달성!")
            elif success_rate >= 60:
                print("좋은 성과를 보이고 있습니다.")
            else:
                print("추가 최적화가 필요합니다.")
            
            if results.get('domain_performance'):
                print(f"\n도메인별 성능:")
                for domain, perf in results['domain_performance'].items():
                    success_rate = (perf['success'] / perf['total']) * 100 if perf['total'] > 0 else 0
                    print(f"  {domain}: {success_rate:.1f}% ({perf['success']}/{perf['total']})")
                    
            # 학습 데이터 현황
            if results.get('learning_data'):
                learning = results['learning_data']
                print(f"\n학습 데이터 현황:")
                print(f"  성공 답변: {learning.get('successful_answers', 0)}개")
                print(f"  실패 답변: {learning.get('failed_answers', 0)}개")
        
        print(f"결과 파일: {output_file.name}")
    except Exception as e:
        print(f"결과 파일: comprehensive_test_result.csv")


def print_multiple_choice_results(results: dict, output_file: Path, test_count: int):
    """객관식 테스트 결과 출력"""
    try:
        print(f"\n=== 객관식 테스트 결과 ===")
        if results.get("success"):
            print(f"처리 완료: {test_count}개 문항")
            success_rate = results.get('success_rate', 0)
            print(f"성공률: {success_rate:.1f}%")
            
            if success_rate >= 80:
                print("객관식 정확도가 높습니다!")
            elif success_rate >= 70:
                print("양호한 객관식 성과입니다.")
            else:
                print("객관식 패턴 분석 보완이 필요합니다.")
        
        print(f"결과 파일: {output_file.name}")
    except Exception as e:
        print(f"결과 파일: 객관식_test_result.csv")


def print_subjective_results(results: dict, output_file: Path, test_count: int):
    """주관식 테스트 결과 출력"""
    try:
        print(f"\n=== 주관식 테스트 결과 ===")
        if results.get("success"):
            print(f"처리 완료: {test_count}개 문항")
            success_rate = results.get('success_rate', 0)
            print(f"성공률: {success_rate:.1f}%")
            
            if success_rate >= 65:
                print("주관식 답변 품질이 우수합니다!")
            elif success_rate >= 50:
                print("주관식 성과가 개선되고 있습니다.")
            else:
                print("주관식 답변 생성 로직 보완이 필요합니다.")
        
        print(f"결과 파일: {output_file.name}")
    except Exception as e:
        print(f"결과 파일: 주관식_test_result.csv")


def print_domain_results(results: dict, output_file: Path, test_count: int, domain_name: str):
    """도메인별 테스트 결과 출력"""
    try:
        print(f"\n=== {domain_name} 도메인 테스트 결과 ===")
        if results.get("success"):
            print(f"처리 완료: {test_count}개 문항")
            success_rate = results.get('success_rate', 0)
            print(f"성공률: {success_rate:.1f}%")
            
            domain_weight = get_domain_weight(domain_name)
            expected_performance = 70 * domain_weight.get("priority_boost", 1.0)
            
            if success_rate >= expected_performance:
                print(f"{domain_name} 도메인 성능이 목표에 도달했습니다!")
            elif success_rate >= expected_performance * 0.8:
                print(f"{domain_name} 도메인이 양호한 성과를 보입니다.")
            else:
                print(f"{domain_name} 도메인 특화 최적화가 필요합니다.")
        
        print(f"결과 파일: {output_file.name}")
    except Exception as e:
        print(f"결과 파일: {domain_name}_domain_test_result.csv")


def print_results(results: dict, output_file: str, test_size: int):
    """기본 결과 출력"""
    try:
        print(f"\n=== 테스트 결과 ===")
        if results.get("success"):
            print(f"처리 완료: {test_size}개 문항")
            success_rate = results.get('success_rate', 0)
            print(f"성공률: {success_rate:.1f}%")
            
            target_rate = 70.0
            if success_rate >= target_rate:
                gap = success_rate - target_rate
                print(f"목표 달성! (+{gap:.1f}%)")
            else:
                gap = target_rate - success_rate
                print(f"목표까지: {gap:.1f}% 부족")
        
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
        "위험관리",
        "정보통신"
    ]
    
    print("\n테스트할 도메인을 선택하세요:")
    for i, domain in enumerate(domains, 1):
        domain_weight = get_domain_weight(domain)
        priority = domain_weight.get("priority_boost", 1.0)
        priority_text = "고우선순위" if priority > 1.2 else "일반"
        print(f"{i}. {domain} ({priority_text})")
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
        options = {"1": 1, "2": 2, "3": 5, "4": 10, "5": "전체"}
        print("1. 1문항 (빠른 테스트)")
        print("2. 2문항 (기본 테스트)")
        print("3. 5문항 (샘플 테스트)")
        print("4. 10문항 (상세 테스트)")
        print("5. 전체 테스트")
    elif test_type == "객관식":
        options = {"1": 5, "2": 10, "3": 50, "4": 100, "5": "전체"}
        print("1. 5문항 (빠른 테스트)")
        print("2. 10문항 (기본 테스트)")
        print("3. 50문항 (표준 테스트)")
        print("4. 100문항 (상세 테스트)")
        print("5. 전체 테스트")
    elif test_type == "종합":
        options = {"1": 8, "2": 15, "3": 50, "4": 100}
        print("1. 8문항 (빠른 종합)")
        print("2. 15문항 (기본 종합)")
        print("3. 50문항 (표준 종합)")
        print("4. 100문항 (상세 종합)")
    elif test_type == "도메인별":
        options = {"1": 5, "2": 10, "3": 20, "4": 50}
        print("1. 5문항 (빠른 도메인)")
        print("2. 10문항 (기본 도메인)")
        print("3. 20문항 (표준 도메인)")
        print("4. 50문항 (상세 도메인)")
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
                
        print("\n테스트가 완료되었습니다.")
        
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"메인 실행 오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()