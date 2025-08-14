# test_runner.py

"""
간단한 테스트 실행기
- 복잡성 제거
- 실제 성능 측정
- 명확한 결과 출력
"""

import os
import sys
import time
from pathlib import Path

# 현재 디렉토리 설정
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from inference import FinancialAIInference

def run_test(test_size: int = 50, verbose: bool = True):
    """테스트 실행"""
    
    print("=" * 60)
    print(f"금융보안 AI 테스트 실행 ({test_size}문항)")
    print("=" * 60)
    
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
        engine = FinancialAIInference(verbose=verbose)
        
        # 테스트 데이터 준비
        import pandas as pd
        test_df = pd.read_csv(test_file)
        submission_df = pd.read_csv(submission_file)
        
        # 지정된 크기로 제한
        if len(test_df) > test_size:
            test_df = test_df.head(test_size)
            submission_df = submission_df.head(test_size)
            
            # 임시 파일 생성
            test_df.to_csv("./test_temp.csv", index=False)
            submission_df.to_csv("./submission_temp.csv", index=False)
            
            output_file = f"./test_result_{test_size}.csv"
            results = engine.execute_inference(
                "./test_temp.csv", 
                "./submission_temp.csv", 
                output_file
            )
            
            # 임시 파일 정리
            os.remove("./test_temp.csv")
            os.remove("./submission_temp.csv")
        else:
            output_file = f"./test_result_{len(test_df)}.csv"
            results = engine.execute_inference(
                test_file,
                submission_file,
                output_file
            )
        
        # 결과 분석
        print_test_results(results, output_file)
        
        return True
        
    except Exception as e:
        print(f"테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if engine:
            engine.cleanup()

def print_test_results(results: dict, output_file: str):
    """테스트 결과 출력"""
    
    print("\n" + "=" * 60)
    print("테스트 결과 분석")
    print("=" * 60)
    
    print(f"처리 완료: {results['total_questions']}문항")
    print(f"객관식: {results['mc_count']}개, 주관식: {results['subj_count']}개")
    print(f"모델 성공률: {results['model_success_rate']:.1f}%")
    print(f"평균 처리시간: {results['avg_processing_time']:.2f}초/문항")
    print(f"총 소요시간: {results['total_time']:.1f}초")
    
    # 객관식 답변 분포 분석
    distribution = results['answer_distribution']
    total_mc = sum(distribution.values())
    
    if total_mc > 0:
        print(f"\n객관식 답변 분포:")
        for num in range(1, 6):
            count = distribution[str(num)]
            pct = (count / total_mc) * 100
            print(f"  {num}번: {count}개 ({pct:.1f}%)")
        
        # 다양성 평가
        used_numbers = len([v for v in distribution.values() if v > 0])
        if used_numbers >= 4:
            diversity_status = "✅ 우수"
        elif used_numbers >= 3:
            diversity_status = "⚠️ 양호"
        else:
            diversity_status = "❌ 개선필요"
        
        print(f"  답변 다양성: {diversity_status} ({used_numbers}/5개 번호 사용)")
    
    # 성능 평가
    print(f"\n성능 평가:")
    if results['model_success_rate'] >= 70:
        print("✅ 모델 성능: 우수")
    elif results['model_success_rate'] >= 50:
        print("⚠️ 모델 성능: 양호")
    else:
        print("❌ 모델 성능: 개선필요")
    
    if results['avg_processing_time'] <= 15:
        print("✅ 처리 속도: 우수")
    elif results['avg_processing_time'] <= 30:
        print("⚠️ 처리 속도: 양호") 
    else:
        print("❌ 처리 속도: 개선필요")
    
    print(f"\n결과 파일: {output_file}")
    
    # 파일 내용 검증
    try:
        import pandas as pd
        result_df = pd.read_csv(output_file)
        
        mc_answers = 0
        subj_answers = 0
        same_answers = 0
        
        for answer in result_df['Answer']:
            if str(answer).strip() in ['1', '2', '3', '4', '5']:
                mc_answers += 1
            else:
                subj_answers += 1
        
        print(f"\n파일 검증:")
        print(f"  객관식 답변: {mc_answers}개")
        print(f"  주관식 답변: {subj_answers}개")
        
        # 기존 문제와 비교
        if mc_answers > 0:
            # 모든 답변이 1번인지 확인
            ones_count = sum(1 for answer in result_df['Answer'] if str(answer).strip() == '1')
            if ones_count == mc_answers:
                print("❌ 문제: 모든 객관식이 1번으로 고정됨")
            else:
                print("✅ 개선: 객관식 답변이 다양함")
        
        if subj_answers > 0:
            # 동일한 템플릿 답변 확인
            template_answer = "체계적인 관리 방안을 수립하고 지속적인 개선을 수행해야 합니다."
            template_count = sum(1 for answer in result_df['Answer'] if str(answer).strip() == template_answer)
            
            if template_count == subj_answers:
                print("❌ 문제: 모든 주관식이 동일한 템플릿")
            else:
                print("✅ 개선: 주관식 답변이 다양함")
        
    except Exception as e:
        print(f"파일 검증 오류: {e}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='금융보안 AI 테스트')
    parser.add_argument('--size', type=int, default=50, help='테스트할 문항 수 (기본: 50)')
    parser.add_argument('--verbose', action='store_true', help='상세 출력')
    
    args = parser.parse_args()
    
    # 테스트 크기 제한
    test_size = max(1, min(args.size, 500))
    
    print(f"Python 버전: {sys.version.split()[0]}")
    print(f"테스트 크기: {test_size}문항")
    
    success = run_test(test_size, args.verbose)
    
    if success:
        print("\n🎉 테스트 완료!")
    else:
        print("\n❌ 테스트 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()