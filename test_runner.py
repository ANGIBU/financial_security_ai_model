# test_runner.py

"""
테스트 실행기
- 시스템 테스트 실행
- 의도 일치 성공률 표시
- 핵심 성능 지표 출력
- LLM 생성 성과 표시
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
        test_size = TEST_CONFIG['default_test_size']
    
    # 파일 존재 확인
    test_file = DEFAULT_FILES['test_file']
    submission_file = DEFAULT_FILES['submission_file']
    
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
        test_df = pd.read_csv(test_file, encoding=FILE_VALIDATION['encoding'])
        submission_df = pd.read_csv(submission_file, encoding=FILE_VALIDATION['encoding'])
        
        print(f"전체 데이터: {len(test_df)}개 문항")
        print(f"테스트 크기: {test_size}개 문항")
        
        # 지정된 크기로 제한
        if len(test_df) > test_size:
            test_df = test_df.head(test_size)
            temp_submission = submission_df.head(test_size).copy()
            
            output_file = DEFAULT_FILES['test_output_file']
            results = engine.execute_inference_with_data(
                test_df, 
                temp_submission, 
                output_file
            )
        else:
            output_file = DEFAULT_FILES['test_output_file']
            results = engine.execute_inference(
                test_file,
                submission_file,
                output_file
            )
        
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

def print_enhanced_results(results: dict, output_file: str, test_size: int):
    """핵심 성과 지표 출력 - LLM 생성 중심"""

    total_time_minutes = results['total_time'] / 60
    print(f"\n=== 테스트 완료 ({test_size}개 문항) ===")
    print(f"처리 시간: {total_time_minutes:.1f}분")
    print(f"처리 문항: {results['total_questions']}개")
    print(f"신뢰도: {results['reliability_score']:.1f}%")
    
    # 기본 성능 지표
    print(f"\n=== 기본 성능 지표 ===")
    print(f"모델 성공률: {results.get('model_success_rate', 0):.1f}%")
    print(f"한국어 준수율: {results.get('korean_compliance_rate', 0):.1f}%")
    print(f"평균 처리시간: {results.get('avg_processing_time', 0):.2f}초")
    
    # LLM 생성 성과
    llm_generation_rate = results.get('llm_generation_rate', 0)
    knowledge_guided_rate = results.get('knowledge_guided_rate', 0)
    
    print(f"\n=== LLM 생성 성과 ===")
    print(f"LLM 생성률: {llm_generation_rate:.1f}%")
    print(f"지식 가이드 생성률: {knowledge_guided_rate:.1f}%")
    print(f"템플릿 힌트 활용률: {results.get('template_hint_rate', 0):.1f}%")
    
    if llm_generation_rate >= 80:
        print("LLM 생성 성과: 우수 (대회 규칙 준수)")
    elif llm_generation_rate >= 60:
        print("LLM 생성 성과: 양호")
    else:
        print("LLM 생성 성과: 개선 필요")
    
    # 객관식 성능
    mc_count = results.get('mc_count', 0)
    mc_context_accuracy = results.get('mc_context_accuracy_rate', 0)
    
    if mc_count > 0:
        print(f"\n=== 객관식 성능 (전체의 {(mc_count/results['total_questions']*100):.0f}%) ===")
        print(f"객관식 문항: {mc_count}개")
        print(f"컨텍스트 정확도: {mc_context_accuracy:.1f}%")
    else:
        print(f"\n=== 객관식 성능 ===")
        print("객관식 문항 없음")
    
    # 주관식 성능
    subj_count = results.get('subj_count', 0)
    intent_success_rate = results.get('intent_match_success_rate', 0)
    avg_quality = results.get('avg_quality_score', 0)
    
    if subj_count > 0:
        print(f"\n=== 주관식 성능 (전체의 {(subj_count/results['total_questions']*100):.0f}%) ===")
        print(f"주관식 문항: {subj_count}개")
        print(f"의도 일치율: {intent_success_rate:.1f}%")
        print(f"평균 품질점수: {avg_quality:.2f}/1.0")
        
        # LLM 생성 품질 평가
        if knowledge_guided_rate > 70:
            print("지식 통합 생성: 우수")
        elif knowledge_guided_rate > 50:
            print("지식 통합 생성: 양호")
        else:
            print("지식 통합 생성: 개선 필요")
    else:
        print(f"\n=== 주관식 성능 ===")
        print("주관식 문항 없음")
    
    # 텍스트 처리 성과
    retry_generation_rate = results.get('retry_generation_rate', 0)
    validation_failure_rate = results.get('validation_failure_rate', 0)
    corruption_detection_rate = results.get('corruption_detection_rate', 0)
    
    print(f"\n=== 텍스트 처리 성과 ===")
    print(f"재생성 요청율: {retry_generation_rate:.1f}%")
    print(f"검증 실패율: {validation_failure_rate:.1f}%")
    print(f"텍스트 깨짐 감지율: {corruption_detection_rate:.1f}%")
    
    # 품질 관리 성과
    print(f"\n=== 품질 관리 성과 ===")
    if retry_generation_rate > 0:
        print(f"품질 자동 관리: {retry_generation_rate:.1f}%의 답변에서 품질 관리 수행")
    if validation_failure_rate < 5:
        print(f"검증 통과율: {100-validation_failure_rate:.1f}% (양호)")
    elif validation_failure_rate < 10:
        print(f"검증 통과율: {100-validation_failure_rate:.1f}% (보통)")
    else:
        print(f"검증 통과율: {100-validation_failure_rate:.1f}% (주의 필요)")
    
    if corruption_detection_rate < 2:
        print("텍스트 안전성: 우수")
    elif corruption_detection_rate < 5:
        print("텍스트 안전성: 양호")
    else:
        print("텍스트 안전성: 관리 필요")
    
    # 성능 분석 요약
    print(f"\n=== 성능 분석 요약 ===")
    
    if results['reliability_score'] >= 80:
        reliability_status = "우수"
    elif results['reliability_score'] >= 70:
        reliability_status = "양호"
    elif results['reliability_score'] >= 60:
        reliability_status = "보통"
    else:
        reliability_status = "관리 필요"
    
    print(f"시스템 신뢰도: {results['reliability_score']:.1f}% ({reliability_status})")
    
    # LLM 생성 평가
    if llm_generation_rate >= 70 and knowledge_guided_rate >= 60:
        print("LLM 생성 시스템: 안정적 (대회 규칙 준수)")
    elif llm_generation_rate >= 50:
        print("LLM 생성 시스템: 보통")
    else:
        print("LLM 생성 시스템: 개선 필요")
    
    if mc_count > 0 and mc_context_accuracy >= 70:
        print("객관식 처리: 안정적")
    elif mc_count > 0:
        print("객관식 처리: 관리 필요")
    
    if subj_count > 0 and intent_success_rate >= 60 and avg_quality >= 0.7:
        print("주관식 처리: 안정적")
    elif subj_count > 0:
        print("주관식 처리: 관리 필요")
    
    # 처리 효율성
    questions_per_minute = results['total_questions'] / total_time_minutes if total_time_minutes > 0 else 0
    print(f"처리 효율성: {questions_per_minute:.1f}문항/분")
    
    # 대회 규칙 준수성 평가
    print(f"\n=== 대회 규칙 준수성 ===")
    compliance_score = 0
    
    if llm_generation_rate >= 80:
        print("✓ LLM 기반 텍스트 생성: 준수")
        compliance_score += 25
    elif llm_generation_rate >= 60:
        print("△ LLM 기반 텍스트 생성: 부분 준수")
        compliance_score += 15
    else:
        print("✗ LLM 기반 텍스트 생성: 미준수")
    
    if knowledge_guided_rate >= 50:
        print("✓ 지식 통합 생성: 준수")
        compliance_score += 25
    else:
        print("△ 지식 통합 생성: 부분 준수")
        compliance_score += 10
    
    if results.get('korean_compliance_rate', 0) >= 95:
        print("✓ 한국어 전용 처리: 준수")
        compliance_score += 25
    elif results.get('korean_compliance_rate', 0) >= 80:
        print("△ 한국어 전용 처리: 부분 준수")
        compliance_score += 15
    else:
        print("✗ 한국어 전용 처리: 미준수")
    
    if validation_failure_rate < 10:
        print("✓ 품질 검증 시스템: 준수")
        compliance_score += 25
    else:
        print("△ 품질 검증 시스템: 부분 준수")
        compliance_score += 10
    
    print(f"\n전체 준수도: {compliance_score}/100점")
    
    # 권장사항
    print(f"\n=== 권장사항 ===")
    recommendations = []
    
    if llm_generation_rate < 70:
        recommendations.append("LLM 생성 비율을 높여 대회 규칙 준수성을 개선하세요.")
    
    if knowledge_guided_rate < 50:
        recommendations.append("지식 가이드 통합을 강화하여 답변 품질을 높이세요.")
    
    if results['reliability_score'] < 70:
        recommendations.append("전체적인 시스템 성능 점검이 필요합니다.")
    
    if mc_count > 0 and mc_context_accuracy < 60:
        recommendations.append("객관식 컨텍스트 분석 정확도를 높이세요.")
    
    if subj_count > 0 and intent_success_rate < 50:
        recommendations.append("주관식 질문 의도 분석 정확도를 높이세요.")
    
    if validation_failure_rate > 10:
        recommendations.append("답변 검증 로직을 점검하세요.")
    
    if retry_generation_rate > 30:
        recommendations.append("초기 답변 생성 품질을 높이세요.")
    
    if corruption_detection_rate > 5:
        recommendations.append("텍스트 안전성 검사를 강화하세요.")
    
    if not recommendations:
        recommendations.append("현재 성능이 양호합니다. LLM 생성 품질을 지속적으로 모니터링하세요.")
    
    for i, recommendation in enumerate(recommendations, 1):
        print(f"{i}. {recommendation}")

def estimate_final_performance(results: dict) -> float:
    """최종 성능 예측 - LLM 생성 중심"""
    
    # 객관식 성과 (가중치 86%)
    mc_weight = 0.86
    mc_context = results.get('mc_context_accuracy_rate', 0) / 100
    
    mc_score = mc_context * mc_weight
    
    # 주관식 성과 (가중치 14%) - LLM 생성 품질 반영
    subj_weight = 0.14
    intent_success = results.get('intent_match_success_rate', 0) / 100
    korean_compliance = results.get('korean_compliance_rate', 0) / 100
    llm_generation = results.get('llm_generation_rate', 0) / 100
    
    # LLM 생성 품질을 반영한 주관식 점수
    subj_score = (intent_success * 0.3 + korean_compliance * 0.3 + llm_generation * 0.4) * subj_weight
    
    # 전체 예상 점수
    total_score = mc_score + subj_score
    
    return total_score

def analyze_llm_generation_performance(results: dict):
    """LLM 생성 성능 분석"""
    print(f"\n=== LLM 생성 성능 분석 ===")
    
    llm_rate = results.get('llm_generation_rate', 0)
    knowledge_rate = results.get('knowledge_guided_rate', 0)
    template_hint_rate = results.get('template_hint_rate', 0)
    retry_rate = results.get('retry_generation_rate', 0)
    
    analyses = []
    
    if llm_rate >= 80:
        analyses.append(f"LLM 생성률이 {llm_rate:.1f}%로 우수하여 대회 규칙을 잘 준수하고 있습니다.")
    elif llm_rate >= 60:
        analyses.append(f"LLM 생성률이 {llm_rate:.1f}%로 양호합니다.")
    else:
        analyses.append(f"LLM 생성률이 {llm_rate:.1f}%로 개선이 필요합니다.")
    
    if knowledge_rate >= 60:
        analyses.append(f"지식 가이드 통합률이 {knowledge_rate:.1f}%로 양호합니다.")
    elif knowledge_rate >= 40:
        analyses.append(f"지식 가이드 통합률이 {knowledge_rate:.1f}%로 보통입니다.")
    else:
        analyses.append(f"지식 가이드 통합률이 {knowledge_rate:.1f}%로 낮습니다.")
    
    if template_hint_rate > 0:
        analyses.append(f"템플릿 힌트가 {template_hint_rate:.1f}%의 경우에 참고용으로 활용되었습니다.")
    
    if retry_rate > 0:
        analyses.append(f"품질 개선을 위한 재생성이 {retry_rate:.1f}%의 경우에 수행되었습니다.")
    
    for i, analysis in enumerate(analyses, 1):
        print(f"{i}. {analysis}")

def select_test_size():
    """테스트 문항 수 선택"""
    print("\n테스트할 문항 수를 선택하세요:")
    
    test_options = TEST_CONFIG['test_sizes']
    print(f"1. {test_options['quick']}문항 (빠른 테스트)")
    print(f"2. {test_options['basic']}문항 (기본 테스트)")
    print(f"3. {test_options['detailed']}문항 (정밀 테스트)")
    print(f"4. {test_options['full']}문항 (전체 테스트)")
    print()
    
    while True:
        try:
            choice = input("선택 (1-4): ").strip()
            
            if choice == "1":
                return test_options['quick']
            elif choice == "2":
                return test_options['basic']
            elif choice == "3":
                return test_options['detailed']
            elif choice == "4":
                return test_options['full']
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
    
    print(f"\n선택된 테스트: {test_size}문항")
    print("LLM 기반 AI 추론 시스템을 테스트합니다...")
    
    success = run_test(test_size, verbose=True)
    
    if success:
        print(f"\n테스트 완료 - 결과 파일: {DEFAULT_FILES['test_output_file']}")
        print("LLM 생성 중심의 추론 시스템이 정상적으로 작동했습니다.")
    else:
        print("\n테스트 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()