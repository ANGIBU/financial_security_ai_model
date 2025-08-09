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
import re
from pathlib import Path
from typing import Dict, List

current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

def test_korean_quality(text: str, question_type: str = "subjective") -> Dict:
    """한국어 품질 검사"""
    
    if question_type == "multiple_choice" or re.match(r'^[1-5]$', text.strip()):
        if re.match(r'^[1-5]$', text.strip()):
            return {
                "has_chinese": False,
                "has_russian": False,
                "chinese_chars": [],
                "russian_chars": [],
                "korean_ratio": 1.0,
                "english_ratio": 0.0,
                "is_good_quality": True
            }
        elif re.search(r'[1-5]', text):
            return {
                "has_chinese": False,
                "has_russian": False,
                "chinese_chars": [],
                "russian_chars": [],
                "korean_ratio": 0.8,
                "english_ratio": 0.0,
                "is_good_quality": True
            }
    
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    russian_chars = re.findall(r'[а-яё]', text.lower())
    
    korean_chars = len(re.findall(r'[가-힣]', text))
    total_chars = len(re.sub(r'[^\w]', '', text))
    korean_ratio = korean_chars / max(total_chars, 1)
    
    english_chars = len(re.findall(r'[A-Za-z]', text))
    english_ratio = english_chars / max(len(text), 1)
    
    is_good_quality = (
        len(chinese_chars) == 0 and 
        len(russian_chars) == 0 and 
        korean_ratio > 0.3 and 
        english_ratio < 0.4
    )
    
    return {
        "has_chinese": len(chinese_chars) > 0,
        "has_russian": len(russian_chars) > 0,
        "chinese_chars": chinese_chars,
        "russian_chars": russian_chars,
        "korean_ratio": korean_ratio,
        "english_ratio": english_ratio,
        "is_good_quality": is_good_quality
    }

def test_basic():
    """기본 시스템 체크"""
    print("기본 시스템 체크")
    print("="*50)
    
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_info.name}")
        print(f"메모리: {gpu_info.total_memory / (1024**3):.1f}GB")
    else:
        print("GPU 없음")
    
    files = ["test.csv", "sample_submission.csv"]
    for f in files:
        if os.path.exists(f):
            print(f"파일 존재: {f}")
            df = pd.read_csv(f)
            print(f"  데이터 수: {len(df)}개")
        else:
            print(f"파일 없음: {f}")
    
    print("\n한국어 처리 모듈 테스트:")
    try:
        from data_processor import DataProcessor
        processor = DataProcessor()
        
        test_texts = [
            ("이것은 软件입니다", "subjective"),
            ("金融 거래의 安全성", "subjective"), 
            ("financial system 관리", "subjective"),
            ("개인정보보호법에 따른 조치", "subjective"),
            ("трой목마 바이러스", "subjective"),
            ("2", "multiple_choice"),
            ("정답: 3", "multiple_choice")
        ]
        
        for text, q_type in test_texts:
            cleaned = processor._clean_korean_text(text)
            quality = test_korean_quality(cleaned, q_type)
            print(f"  원본: {text}")
            print(f"  정리: {cleaned}")
            print(f"  유형: {q_type}")
            print(f"  품질: {'양호' if quality['is_good_quality'] else '개선필요'}")
            if quality['has_chinese']:
                print(f"    한자 발견: {quality['chinese_chars']}")
            if quality['has_russian']:
                print(f"    러시아어 발견: {quality['russian_chars']}")
            print()
            
    except Exception as e:
        print(f"  오류: {e}")
    
    print()

def test_korean_generation():
    """한국어 생성 테스트"""
    print("한국어 생성 품질 테스트")
    print("="*50)
    
    try:
        from model_handler import ModelHandler
        
        print("모델 로딩 중...")
        start = time.time()
        
        handler = ModelHandler(
            model_name="upstage/SOLAR-10.7B-Instruct-v1.0",
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False
        )
        load_time = time.time() - start
        print(f"모델 로딩 시간: {load_time:.1f}초")
        
        test_prompts = [
            {
                "name": "객관식 테스트",
                "prompt": """### 지시사항
당신은 한국의 금융보안 전문가입니다.
반드시 한국어로만 답변하고, 한자나 영어는 절대 사용하지 마세요.

### 문제
개인정보보호법상 개인정보의 정의로 가장 적절한 것은?
1. 모든 정보
2. 살아 있는 개인에 관한 정보로서 개인을 알아볼 수 있는 정보
3. 기업 정보
4. 공개된 정보
5. 암호화된 정보

반드시 한국어로만 간단히 분석한 후 정답 번호만 제시하세요.

정답 번호:""",
                "type": "multiple_choice"
            },
            {
                "name": "주관식 테스트",
                "prompt": """### 지시사항
당신은 한국의 금융보안 전문가입니다.
반드시 한국어로만 답변하세요. 한자, 영어 등 모든 외국어는 절대 사용하지 마세요.

### 질문
개인정보보호 관리체계 구축 방안을 설명하세요.

### 중요 사항
- 전문적이고 정확한 한국어만 사용
- 한자나 영어 단어 사용 금지
- 법령명도 한국어로 표기

답변:""",
                "type": "subjective"
            },
            {
                "name": "트로이 목마 테스트",
                "prompt": """### 지시사항
당신은 한국의 사이버보안 전문가입니다.
반드시 순수 한국어로만 답변하세요.

### 질문
트로이 목마 기반 원격제어 악성코드의 특징과 탐지 지표를 설명하세요.

답변:""",
                "type": "subjective"
            }
        ]
        
        print("\n한국어 생성 테스트...")
        for test in test_prompts:
            print(f"\n{test['name']}")
            print("-" * 30)
            
            start_time = time.time()
            result = handler.generate_response(
                test["prompt"], 
                test["type"],
                max_attempts=1
            )
            elapsed = time.time() - start_time
            
            print(f"응답 시간: {elapsed:.2f}초")
            print(f"원본 응답: {result.response}")
            
            quality = test_korean_quality(result.response, test["type"])
            print(f"\n품질 분석:")
            print(f"  한자 포함: {'예' if quality['has_chinese'] else '아니오'}")
            print(f"  러시아어 포함: {'예' if quality['has_russian'] else '아니오'}")
            if quality['chinese_chars']:
                print(f"  한자 문자: {quality['chinese_chars']}")
            if quality['russian_chars']:
                print(f"  러시아어 문자: {quality['russian_chars']}")
            print(f"  한국어 비율: {quality['korean_ratio']:.2%}")
            print(f"  영어 비율: {quality['english_ratio']:.2%}")
            print(f"  품질 평가: {'우수' if quality['is_good_quality'] else '개선필요'}")
        
        handler.cleanup()
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()

def test_small_inference_clean(sample_size: int = 5, with_learning: bool = False, verbose: bool = False):
    """실제 추론 엔진을 사용한 소규모 테스트"""
    mode_text = "학습 포함" if with_learning else "기본"
    print(f"소규모 추론 테스트 ({sample_size}개, {mode_text} 모드)")
    print("="*60)
    
    if not os.path.exists("test.csv"):
        print("오류: test.csv 파일이 없습니다.")
        return
    
    from inference import FinancialAIInference
    
    test_df = pd.read_csv("test.csv")
    
    if len(test_df) < sample_size:
        sample_size = len(test_df)
        print(f"데이터 부족. {sample_size}개만 테스트합니다.")
    
    sample_df = test_df.head(sample_size).copy()
    
    engine = None
    try:
        print(f"\n모델 로딩 중...")
        engine = FinancialAIInference(enable_learning=with_learning, verbose=verbose)
        
        answers = []
        korean_quality_results = []
        question_types = []
        
        print(f"\n추론 시작...")
        
        for idx, row in sample_df.iterrows():
            question = row['Question']
            question_id = row['ID']
            
            structure = engine.data_processor.analyze_question_structure(question)
            question_type = structure["question_type"]
            question_types.append(question_type)
            
            answer = engine.process_question(question, question_id, idx)
            answers.append(answer)
            
            question_short = question[:80] + "..." if len(question) > 80 else question
            
            print(f"\n문제 {idx+1}: {question_short}")
            print(f"문제 유형: {question_type}")
            print(f"답변: {answer}")
            
            if verbose and question_type == "subjective":
                print(f"답변 전문: {answer}")
            
            quality = test_korean_quality(answer, question_type)
            korean_quality_results.append(quality)
        
        mc_answers = []
        subj_answers = []
        mc_qualities = []
        subj_qualities = []
        
        for answer, q_type, quality in zip(answers, question_types, korean_quality_results):
            if q_type == "multiple_choice":
                mc_answers.append(answer)
                mc_qualities.append(quality)
            else:
                subj_answers.append(answer)
                subj_qualities.append(quality)
        
        total_good_quality = sum(1 for q in korean_quality_results if q['is_good_quality'])
        total_with_foreign = sum(1 for q in korean_quality_results if q['has_chinese'] or q['has_russian'])
        
        all_korean_ratios = [q['korean_ratio'] for q in korean_quality_results]
        avg_korean_ratio = sum(all_korean_ratios) / len(all_korean_ratios) if all_korean_ratios else 0
        
        print("\n" + "="*60)
        print("테스트 결과")
        print("="*60)
        
        print(f"\n문제 유형 분포:")
        print(f"  객관식: {len(mc_answers)}개")
        print(f"  주관식: {len(subj_answers)}개")
        
        print(f"\n한국어 품질 리포트:")
        print(f"  외국어 포함 답변: {total_with_foreign}/{sample_size}개")
        print(f"  품질 우수 답변: {total_good_quality}/{sample_size}개")
        print(f"  평균 한국어 비율: {avg_korean_ratio:.2%}")
        print(f"  전체 품질 점수: {(total_good_quality/sample_size)*100:.1f}점")
        
        if total_with_foreign == 0:
            print("  외국어 혼재 문제 해결됨")
        else:
            print("  일부 답변에 외국어 포함")
            for i, quality in enumerate(korean_quality_results):
                if quality['has_chinese'] or quality['has_russian']:
                    print(f"    문제 {i+1}: 외국어 발견")
        
        if with_learning:
            print(f"\n학습 통계:")
            print(f"  학습된 샘플: {engine.stats.get('learned', 0)}개")
            print(f"  한국어 수정: {engine.stats.get('korean_fixes', 0)}회")
            if hasattr(engine, 'learning_system'):
                print(f"  정확도: {engine.learning_system.get_current_accuracy():.2%}")
        
        if mc_answers:
            mc_distribution = {}
            for ans in mc_answers:
                mc_distribution[ans] = mc_distribution.get(ans, 0) + 1
            
            print(f"\n객관식 답변 분포 ({len(mc_answers)}개):")
            for num in sorted(mc_distribution.keys()):
                count = mc_distribution[num]
                pct = (count / len(mc_answers)) * 100
                print(f"  {num}번: {count}개 ({pct:.1f}%)")
            
            unique_mc = len(mc_distribution)
            if unique_mc >= min(4, len(mc_answers)):
                print("  객관식 답변 다양성 양호")
            else:
                print("  객관식 답변 다양성 부족")
        
        if subj_answers:
            print(f"\n주관식 답변 품질 ({len(subj_answers)}개):")
            for i, (ans, quality) in enumerate(zip(subj_answers, subj_qualities)):
                print(f"  답변 {i+1}: {ans[:50]}...")
                print(f"    품질: {'우수' if quality['is_good_quality'] else '개선필요'}")
                if quality['has_chinese'] or quality['has_russian']:
                    print(f"    외국어 발견")
                if verbose:
                    print(f"    전문: {ans}")
        
        print(f"\n추가 검증:")
        
        misclassified = 0
        for i, (q_type, answer) in enumerate(zip(question_types, answers)):
            question = sample_df.iloc[i]['Question'].lower()
            if ("설명하세요" in question or "기술하세요" in question or "트로이" in question) and q_type == "multiple_choice":
                misclassified += 1
                print(f"  문제 {i+1}: 주관식이 객관식으로 분류됨")
        
        if misclassified == 0:
            print("  문제 유형 분류 정확")
        else:
            print(f"  문제 유형 오분류: {misclassified}개")
        
        wrong_mc_answers = 0
        for i, (q_type, answer) in enumerate(zip(question_types, answers)):
            if q_type == "multiple_choice":
                if not (answer.isdigit() and 1 <= int(answer) <= 5):
                    wrong_mc_answers += 1
                    print(f"  문제 {i+1}: 객관식 답변 형식 오류 - '{answer}'")
        
        wrong_subj_answers = 0
        for i, (q_type, answer) in enumerate(zip(question_types, answers)):
            if q_type == "subjective":
                if re.match(r'^[1-5]$', answer.strip()):
                    wrong_subj_answers += 1
                    print(f"  문제 {i+1}: 주관식에 객관식 답변 - '{answer}'")
        
        if wrong_mc_answers == 0 and wrong_subj_answers == 0:
            print("  답변 형식 정확")
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if engine:
            engine.cleanup()

def test_small_inference_korean(sample_size: int = 5, with_learning: bool = False, verbose: bool = False):
    """기존 인터페이스 유지용 래퍼"""
    test_small_inference_clean(sample_size, with_learning, verbose)

def test_speed():
    """속도 테스트"""
    print("속도 테스트")
    print("="*50)
    
    from model_handler import ModelHandler
    
    print("모델 로딩 중...")
    start = time.time()
    
    try:
        handler = ModelHandler(
            model_name="upstage/SOLAR-10.7B-Instruct-v1.0",
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False
        )
        load_time = time.time() - start
        print(f"모델 로딩 시간: {load_time:.1f}초")
        
        test_prompt = """### 지시사항
당신은 한국의 금융보안 전문가입니다.
반드시 한국어로만 답변하고, 한자나 영어는 절대 사용하지 마세요.

### 문제
개인정보보호법상 개인정보의 정의로 가장 적절한 것은?
1. 모든 정보
2. 살아 있는 개인에 관한 정보로서 개인을 알아볼 수 있는 정보
3. 기업 정보
4. 공개된 정보
5. 암호화된 정보

반드시 한국어로만 간단히 분석한 후 정답 번호만 제시하세요.

정답 번호:"""
        
        print("\n추론 테스트...")
        
        _ = handler.generate_response(test_prompt, "multiple_choice", max_attempts=1)
        
        times = []
        korean_quality_scores = []
        
        for i in range(3):
            start = time.time()
            result = handler.generate_response(test_prompt, "multiple_choice", max_attempts=1)
            elapsed = time.time() - start
            times.append(elapsed)
            
            quality = test_korean_quality(result.response, "multiple_choice")
            korean_quality_scores.append(quality['is_good_quality'])
            
            print(f"시도 {i+1}: {elapsed:.2f}초")
            print(f"  답변: {result.response[:50]}")
            print(f"  한국어 품질: {'양호' if quality['is_good_quality'] else '개선필요'}")
            if quality['has_chinese'] or quality['has_russian']:
                print(f"  외국어 발견")
        
        avg_time = sum(times) / len(times)
        korean_success_rate = sum(korean_quality_scores) / len(korean_quality_scores)
        
        print(f"\n성능 결과")
        print("="*30)
        print(f"평균 추론 시간: {avg_time:.2f}초")
        print(f"한국어 품질 성공률: {korean_success_rate:.2%}")
        print(f"예상 전체 시간 (515문항): {avg_time * 515 / 60:.1f}분")
        
        if korean_success_rate >= 0.9:
            print("한국어 품질 우수")
        else:
            print("한국어 품질 개선 필요")
        
        handler.cleanup()
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()

def show_menu():
    """메뉴 표시"""
    print("실행 옵션:")
    print("1. 기본 시스템 체크")
    print("2. 한국어 생성 품질 테스트")
    print("3. 소규모 추론 테스트 (5문항, 간소 로그)")
    print("4. 중규모 추론 테스트 (10문항, 간소 로그)")
    print("5. 상세 로그 추론 테스트 (5문항)")
    print("6. 학습 포함 추론 테스트 (5문항)")
    print("7. 속도 성능 테스트")
    print("8. 전체 추론 실행 (515문항)")
    print("0. 종료")
    print("-"*50)

def interactive_mode():
    """대화형 모드"""
    show_menu()
    
    try:
        choice = input("\n선택하세요 (0-8): ").strip()
        
        if choice == "0":
            print("종료합니다.")
            return
        elif choice == "1":
            test_basic()
        elif choice == "2":
            test_korean_generation()
        elif choice == "3":
            test_small_inference_clean(5, with_learning=False, verbose=False)
        elif choice == "4":
            test_small_inference_clean(10, with_learning=False, verbose=False)
        elif choice == "5":
            test_small_inference_clean(5, with_learning=False, verbose=True)
        elif choice == "6":
            test_small_inference_clean(5, with_learning=True, verbose=False)
        elif choice == "7":
            test_speed()
        elif choice == "8":
            confirm = input("전체 515문항을 실행합니다. 계속하시겠습니까? (y/n): ")
            if confirm.lower() == 'y':
                os.system("python inference.py")
            else:
                print("취소되었습니다.")
        else:
            print("잘못된 선택입니다.")
            
    except KeyboardInterrupt:
        print("\n\n종료합니다.")
    except Exception as e:
        print(f"오류: {e}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="금융 AI 테스트")
    parser.add_argument("--test-type", choices=["basic", "korean", "small", "speed", "interactive"], 
                       help="테스트 유형")
    parser.add_argument("--sample-size", type=int, default=5, 
                       help="샘플 크기 (small 테스트용)")
    parser.add_argument("--with-learning", action="store_true",
                       help="학습 기능 활성화")
    parser.add_argument("--verbose", action="store_true",
                       help="상세 로그 출력")
    
    args = parser.parse_args()
    
    if not args.test_type:
        interactive_mode()
    elif args.test_type == "basic":
        test_basic()
    elif args.test_type == "korean":
        test_korean_generation()
    elif args.test_type == "small":
        test_small_inference_clean(args.sample_size, args.with_learning, args.verbose)
    elif args.test_type == "speed":
        test_speed()
    elif args.test_type == "interactive":
        interactive_mode()

if __name__ == "__main__":
    main()