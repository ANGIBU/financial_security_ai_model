# test_runner.py

"""
테스트 실행기
- 시스템 테스트 실행
- 성능 측정 및 분석
- 결과 검증
- 의도 일치 성공률 표시 추가
"""

import os
import sys
from pathlib import Path

# 현재 디렉토리 설정
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from inference import FinancialAIInference

def run_test(test_size: int = 50, verbose: bool = True):
    """테스트 실행"""
    
    # 파일 존재 확인
    test_file = "./test.csv"
    submission_file = "./sample_submission.csv"
    
    for file_path in [test_file, submission_file]:
        if not os.path.exists(file_path):
            print(f"오류: {file_path} 파일이 없습니다")
            return False
    
    engine = None
    try:
        # AI 엔진 초기화
        print("\n시스템 초기화 중...")
        engine = FinancialAIInference(verbose=False)  # 초기화시 verbose=False로 설정
        
        # 테스트 데이터 준비
        import pandas as pd
        test_df = pd.read_csv(test_file)
        submission_df = pd.read_csv(submission_file)
        
        # 지정된 크기로 제한
        if len(test_df) > test_size:
            test_df = test_df.head(test_size)
            temp_submission = submission_df.head(test_size).copy()
            
            # 통일된 출력 파일명
            output_file = "./test_result.csv"
                
            results = engine.execute_inference_with_data(
                test_df, 
                temp_submission, 
                output_file
            )
        else:
            output_file = "./test_result.csv"
            results = engine.execute_inference(
                test_file,
                submission_file,
                output_file
            )
        
        # 결과 분석
        print_test_results(results, output_file, test_size)
        
        return True
        
    except Exception as e:
        print(f"테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if engine:
            engine.cleanup()

def print_test_results(results: dict, output_file: str, test_size: int):
    """테스트 결과 출력 (강화)"""

    print("테스트 완료")
    
    # 기본 정보
    total_time_minutes = results['total_time'] / 60
    
    print(f"\n기본 정보:")
    print(f"  처리 시간: {total_time_minutes:.1f}분")
    print(f"  처리 문항: {results['total_questions']}개")
    print(f"  결과 파일: {output_file}")
    
    # 성능 지표 (강화)
    print(f"\n성능 지표:")
    print(f"  모델 성공률: {results['model_success_rate']:.1f}%")
    print(f"  한국어 준수율: {results['korean_compliance_rate']:.1f}%")
    
    # 의도 일치 성공률 표시 (신규)
    if results.get('intent_match_success_rate', 0) > 0:
        print(f"  의도 일치 성공률: {results['intent_match_success_rate']:.1f}%")
    
    # 오류율 정보
    if results.get('choice_range_error_rate', 0) > 0:
        print(f"  선택지 범위 오류율: {results['choice_range_error_rate']:.1f}%")
    
    if results.get('validation_error_rate', 0) > 0:
        print(f"  검증 오류율: {results['validation_error_rate']:.1f}%")
    
    # 추가 분석 정보
    if results.get('template_usage_rate', 0) > 0:
        print(f"  템플릿 사용률: {results['template_usage_rate']:.1f}%")
    
    if results.get('institution_questions_count', 0) > 0:
        print(f"  기관 관련 질문: {results['institution_questions_count']}개")
    
    # 의도별 품질 정보 (신규)
    intent_quality = results.get('intent_quality_by_type', {})
    if intent_quality:
        print(f"\n의도별 답변 품질:")
        for intent, quality in intent_quality.items():
            print(f"    {intent}: {quality:.2f} (평균)")
    
    # 문항 유형별 분포
    mc_count = results.get('mc_count', 0)
    subj_count = results.get('subj_count', 0)
    if mc_count > 0 or subj_count > 0:
        print(f"\n문항 유형별 분포:")
        print(f"  객관식: {mc_count}개")
        print(f"  주관식: {subj_count}개")
    
    # 도메인별 분포
    domain_stats = results.get('domain_stats', {})
    if domain_stats:
        print(f"\n도메인별 분포:")
        for domain, count in domain_stats.items():
            print(f"    {domain}: {count}개")
    
    # 성능 개선 제안 (신규)
    print_performance_suggestions(results)

def print_performance_suggestions(results: dict):
    """성능 개선 제안 출력 (신규)"""
    suggestions = []
    
    # 의도 일치 성공률 체크
    intent_rate = results.get('intent_match_success_rate', 0)
    if intent_rate < 60:
        suggestions.append(f"의도 일치 성공률이 {intent_rate:.1f}%로 낮습니다. 질문 의도 분석 개선이 필요합니다.")
    
    # 템플릿 사용률 체크
    template_rate = results.get('template_usage_rate', 0)
    if template_rate > 20:
        suggestions.append(f"템플릿 사용률이 {template_rate:.1f}%로 높습니다. 모델 답변 품질 개선이 필요합니다.")
    
    # 검증 오류율 체크
    validation_error_rate = results.get('validation_error_rate', 0)
    if validation_error_rate > 5:
        suggestions.append(f"검증 오류율이 {validation_error_rate:.1f}%로 높습니다. 답변 검증 로직 개선이 필요합니다.")
    
    # 선택지 범위 오류율 체크
    choice_error_rate = results.get('choice_range_error_rate', 0)
    if choice_error_rate > 2:
        suggestions.append(f"선택지 범위 오류율이 {choice_error_rate:.1f}%입니다. 객관식 답변 처리 개선이 필요합니다.")
    
    # 평균 품질 점수 체크
    avg_quality = results.get('avg_quality_score', 0)
    if avg_quality < 0.7:
        suggestions.append(f"평균 답변 품질이 {avg_quality:.2f}로 낮습니다. 답변 생성 품질 개선이 필요합니다.")
    
    if suggestions:
        print(f"\n성능 개선 제안:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    else:
        print(f"\n성능 상태: 양호")

def select_test_size():
    """테스트 문항 수 선택"""
    print("\n테스트할 문항 수를 선택하세요:")
    print("1. 5문항 (빠른 테스트)")
    print("2. 10문항 (기본 테스트)")
    print("3. 50문항 (정밀 테스트)")
    print("4. 100문항 (성능 테스트)")
    print()
    
    while True:
        try:
            choice = input("선택 (1-4): ").strip()
            
            if choice == "1":
                return 5
            elif choice == "2":
                return 10
            elif choice == "3":
                return 50
            elif choice == "4":
                return 100
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
    print("한국어 전용 답변 모드로 실행됩니다.")
    print("의도 일치 성공률 및 상세 성능 분석이 제공됩니다.")
    
    success = run_test(test_size, verbose=True)
    
    if success:
        print("\n테스트가 완료되었습니다.")
        print("성능 개선이 필요한 부분이 있다면 위의 제안사항을 참고하세요.")
    else:
        print("\n테스트 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()