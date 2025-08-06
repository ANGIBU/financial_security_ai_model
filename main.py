# main.py
"""
개발용 메인 파일
"""

import os
import sys
import time
import pandas as pd
import torch
import argparse
from pathlib import Path

# 현재 디렉토리 추가
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

def test_basic():
    """기본 테스트"""
    print("="*50)
    print("기본 시스템 체크")
    print("="*50)
    
    # GPU 확인
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_info.name}")
        print(f"메모리: {gpu_info.total_memory / (1024**3):.1f}GB")
    else:
        print("GPU 없음")
    
    # 파일 확인
    files = ["test.csv", "sample_submission.csv"]
    for f in files:
        if os.path.exists(f):
            print(f"✓ {f} 존재")
            df = pd.read_csv(f)
            print(f"  - {len(df)}개 데이터")
        else:
            print(f"✗ {f} 없음")
    
    print()

def test_small_inference(sample_size: int = 5):
    """소규모 추론 테스트"""
    print("="*50)
    print(f"소규모 추론 테스트 ({sample_size}개)")
    print("="*50)
    
    # 파일 확인
    if not os.path.exists("test.csv"):
        print("오류: test.csv 파일이 없습니다.")
        return
    
    from inference import FinancialAIInference
    
    # 데이터 로드
    test_df = pd.read_csv("test.csv")
    
    if len(test_df) < sample_size:
        sample_size = len(test_df)
        print(f"데이터 부족. {sample_size}개만 테스트합니다.")
    
    sample_df = test_df.head(sample_size).copy()
    
    # 임시 파일 생성
    sample_df.to_csv("test_sample.csv", index=False)
    
    # 추론 실행
    engine = None
    try:
        start = time.time()
        print("\n모델 로딩 중...")
        engine = FinancialAIInference()
        
        answers = []
        for idx, row in sample_df.iterrows():
            question = row['Question']
            question_id = row['ID']
            
            print(f"\n문제 {idx+1}/{sample_size}:")
            print(f"ID: {question_id}")
            print(f"질문: {question[:100]}...")
            
            answer = engine.process_question(question, question_id, idx)
            answers.append(answer)
            
            if len(answer) > 100:
                print(f"답변: {answer[:100]}...")
            else:
                print(f"답변: {answer}")
        
        elapsed = time.time() - start
        
        # 결과 분석
        mc_answers = [a for a in answers if a.isdigit()]
        subj_answers = [a for a in answers if not a.isdigit()]
        
        print("\n" + "="*50)
        print("테스트 결과")
        print("="*50)
        print(f"처리 시간: {elapsed:.1f}초")
        print(f"평균 시간: {elapsed/sample_size:.1f}초/문항")
        
        if mc_answers:
            print(f"\n객관식 답변 ({len(mc_answers)}개): {mc_answers}")
            # 분포 확인
            distribution = {}
            for ans in mc_answers:
                distribution[ans] = distribution.get(ans, 0) + 1
            print("답변 분포:", distribution)
        
        if subj_answers:
            print(f"\n주관식 답변 ({len(subj_answers)}개)")
            for i, ans in enumerate(subj_answers[:3]):  # 최대 3개만 표시
                print(f"  {i+1}. {ans[:50]}...")
        
        # 전체 예상 시간
        total_questions = len(test_df)
        estimated_time = (elapsed / sample_size) * total_questions / 60
        print(f"\n예상 전체 시간: {estimated_time:.1f}분")
        
        if estimated_time > 270:
            print("⚠️ 시간 제한 초과 예상")
        else:
            print("✓ 시간 제한 내 완료 가능")
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if engine:
            engine.cleanup()
        
        # 임시 파일 삭제
        if os.path.exists("test_sample.csv"):
            os.remove("test_sample.csv")

def test_patterns():
    """패턴 테스트"""
    print("="*50)
    print("패턴 학습 테스트")
    print("="*50)
    
    from pattern_learner import AnswerPatternLearner
    from data_processor import DataProcessor
    
    # 샘플 문제들
    test_questions = [
        "개인정보란 살아 있는 개인에 관한 정보로서 다음 중 개인정보의 정의로 가장 적절한 것은?\n1. 모든 정보\n2. 살아 있는 개인에 관한 정보로서 개인을 알아볼 수 있는 정보\n3. 기업 정보\n4. 공개된 정보\n5. 암호화된 정보",
        "전자금융거래란 전자적 장치를 통하여 금융상품과 서비스를 제공하고 이용하는 거래를 말한다. 다음 중 전자금융거래의 정의로 옳은 것은?\n1. 모든 거래\n2. 전자적 장치를 통한 금융상품 및 서비스 거래\n3. 인터넷만\n4. ATM만\n5. 현금거래",
        "다음 중 개인정보 유출 시 조치사항으로 적절하지 않은 것은?\n1. 즉시 통지\n2. 1주일 후 통지\n3. 개인정보보호위원회 신고\n4. 피해 최소화 조치\n5. 재발 방지 대책",
    ]
    
    learner = AnswerPatternLearner()
    processor = DataProcessor()
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n테스트 {i}:")
        print(f"질문: {question[:80]}...")
        
        structure = processor.analyze_question_structure(question)
        pattern = learner.analyze_question_pattern(question)
        
        if pattern:
            print(f"매칭 패턴: {pattern['rule']}")
            print(f"신뢰도: {pattern['base_confidence']:.2f}")
            
            answer, confidence = learner.predict_answer(question, structure)
            print(f"예측 답: {answer}번 (신뢰도: {confidence:.2f})")
        else:
            print("패턴 매칭 없음")
            answer, confidence = learner.predict_answer(question, structure)
            print(f"통계적 예측: {answer}번 (신뢰도: {confidence:.2f})")

def test_speed():
    """속도 테스트"""
    print("="*50)
    print("속도 테스트")
    print("="*50)
    
    from model_handler import ModelHandler
    
    print("모델 로딩 중...")
    start = time.time()
    
    try:
        handler = ModelHandler(
            model_name="upstage/SOLAR-10.7B-Instruct-v1.0",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        load_time = time.time() - start
        print(f"모델 로딩 시간: {load_time:.1f}초")
        
        # 테스트 프롬프트
        test_prompt = """### 문제
개인정보보호법상 개인정보의 정의로 가장 적절한 것은?
1. 모든 정보
2. 살아 있는 개인에 관한 정보로서 개인을 알아볼 수 있는 정보
3. 기업 정보
4. 공개된 정보
5. 암호화된 정보

### 정답
정답 번호는"""
        
        print("\n추론 테스트...")
        
        # Warm-up
        _ = handler.generate_response(test_prompt, "multiple_choice", max_attempts=1)
        
        # 실제 측정
        times = []
        for i in range(3):
            start = time.time()
            result = handler.generate_response(test_prompt, "multiple_choice", max_attempts=1)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"시도 {i+1}: {elapsed:.2f}초 - 답변: {result.response[:50]}...")
        
        avg_time = sum(times) / len(times)
        print(f"\n평균 추론 시간: {avg_time:.2f}초")
        print(f"예상 전체 시간 (515문항): {avg_time * 515 / 60:.1f}분")
        
        handler.cleanup()
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()

def show_menu():
    """메뉴 표시"""
    print("\n" + "="*50)
    print("금융 AI Challenge 테스트 시스템")
    print("="*50)
    print("\n실행 옵션:")
    print("1. 기본 시스템 체크")
    print("2. 소규모 추론 테스트 (5문항)")
    print("3. 중규모 추론 테스트 (10문항)")
    print("4. 패턴 학습 테스트")
    print("5. 속도 성능 테스트")
    print("6. 전체 추론 실행 (515문항)")
    print("0. 종료")
    print("-"*50)

def interactive_mode():
    """대화형 모드"""
    while True:
        show_menu()
        
        try:
            choice = input("\n선택하세요 (0-6): ").strip()
            
            if choice == "0":
                print("종료합니다.")
                break
            elif choice == "1":
                test_basic()
            elif choice == "2":
                test_small_inference(5)
            elif choice == "3":
                test_small_inference(10)
            elif choice == "4":
                test_patterns()
            elif choice == "5":
                test_speed()
            elif choice == "6":
                confirm = input("전체 515문항을 실행합니다. 계속하시겠습니까? (y/n): ")
                if confirm.lower() == 'y':
                    os.system("python inference.py")
                else:
                    print("취소되었습니다.")
            else:
                print("잘못된 선택입니다.")
                
        except KeyboardInterrupt:
            print("\n\n종료합니다.")
            break
        except Exception as e:
            print(f"오류: {e}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="금융 AI 테스트")
    parser.add_argument("--test-type", choices=["basic", "small", "patterns", "speed", "interactive"], 
                       help="테스트 유형")
    parser.add_argument("--sample-size", type=int, default=5, 
                       help="샘플 크기 (small 테스트용)")
    
    args = parser.parse_args()
    
    # 인자가 없으면 대화형 모드
    if not args.test_type:
        interactive_mode()
    elif args.test_type == "basic":
        test_basic()
    elif args.test_type == "small":
        test_small_inference(args.sample_size)
    elif args.test_type == "patterns":
        test_patterns()
    elif args.test_type == "speed":
        test_speed()
    elif args.test_type == "interactive":
        interactive_mode()

if __name__ == "__main__":
    main()