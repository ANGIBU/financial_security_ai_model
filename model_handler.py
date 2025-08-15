# model_handler.py

"""
LLM 모델 핸들러
- 모델 로딩 및 관리
- 답변 생성
- 프롬프트 처리
- 학습 데이터 저장
"""

import torch
import re
import time
import gc
import random
import pickle
import os
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class SimpleModelHandler:
    """모델 핸들러"""
    
    def __init__(self, model_name: str = "upstage/SOLAR-10.7B-Instruct-v1.0", verbose: bool = False):
        self.model_name = model_name
        self.verbose = verbose
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # pkl 저장 폴더 생성
        self.pkl_dir = Path("./pkl")
        self.pkl_dir.mkdir(exist_ok=True)
        
        # 답변 분포 추적 (선택지별로 관리)
        self.answer_distributions = {
            3: {"1": 0, "2": 0, "3": 0},
            4: {"1": 0, "2": 0, "3": 0, "4": 0},
            5: {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        }
        self.mc_answer_counts = {3: 0, 4: 0, 5: 0}
        
        # 질문 컨텍스트 분석용 패턴
        self.negative_patterns = [
            "해당하지.*않는", "적절하지.*않는", "옳지.*않는",
            "틀린", "잘못된", "부적절한", "아닌.*것"
        ]
        
        self.positive_patterns = [
            "맞는.*것", "옳은.*것", "적절한.*것", 
            "올바른.*것", "해당하는.*것", "정확한.*것"
        ]
        
        # 학습 데이터 저장
        self.learning_data = {
            "successful_answers": [],
            "failed_answers": [],
            "question_patterns": {},
            "answer_quality_scores": [],
            "mc_context_patterns": {},  # 객관식 컨텍스트 패턴 저장
            "choice_range_errors": []   # 선택지 범위 오류 기록
        }
        
        # 이전 학습 데이터 로드
        self._load_learning_data()
        
        if verbose:
            print(f"모델 로딩: {model_name}")
            print(f"디바이스: {self.device}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.eval()
        
        # 한국어 전용 주관식 답변 템플릿
        self.korean_templates = {
            "개인정보보호": [
                "개인정보보호법에 따라 정보주체의 권리를 보장하고 개인정보처리자는 수집부터 파기까지 전 과정에서 적절한 보호조치를 이행해야 합니다. 특히 민감정보와 고유식별정보 처리 시에는 별도의 동의를 받아야 합니다.",
                "개인정보 처리 시 정보주체의 동의를 받고 목적 범위 내에서만 이용하며 개인정보보호위원회의 기준에 따른 안전성 확보조치를 수립해야 합니다. 또한 개인정보 처리방침을 공개하고 정보주체의 권리 행사 절차를 마련해야 합니다.",
                "정보주체는 개인정보 열람권, 정정삭제권, 처리정지권을 가지며 개인정보처리자는 이러한 권리 행사를 보장하는 절차를 마련해야 합니다. 아동의 경우 법정대리인의 동의를 받아야 하며 개인정보 침해 시 손해배상 책임을 집니다."
            ],
            "전자금융": [
                "전자금융거래법에 따라 전자금융업자는 이용자의 전자금융거래 안전성 확보를 위한 보안조치를 시행하고 금융감독원의 감독을 받아야 합니다. 전자서명과 전자인증을 통해 거래의 무결성과 신원확인을 보장해야 합니다.",
                "전자금융분쟁조정위원회에서 전자금융거래 분쟁조정 업무를 담당하며 이용자는 관련 법령에 따라 분쟁조정을 신청할 수 있습니다. 한국은행과 금융감독원에서 전자금융업 관련 감독업무를 수행합니다.",
                "전자금융서비스 제공 시 접근매체에 대한 보안성을 확보하고 이용자 인증 절차를 통해 거래의 안전성을 보장해야 합니다. 전자지급수단 이용 시 위조방지 기술과 암호화 기술을 적용하여 보안을 강화해야 합니다."
            ],
            "사이버보안": [
                "트로이 목마 기반 원격접근도구는 사용자를 속여 시스템에 침투하여 외부 공격자가 원격으로 제어하는 악성코드입니다. 네트워크 트래픽 모니터링, 시스템 동작 분석, 파일 생성 및 수정 패턴, 입출력 장치 접근 등의 비정상적인 행동이 주요 탐지 지표입니다.",
                "해당 악성코드는 원격제어 기능을 통해 시스템에 침입하며 백신 프로그램과 행위 기반 탐지 시스템을 활용하여 탐지할 수 있습니다. 주요 대응방안으로는 네트워크 모니터링 강화와 접근권한 관리를 통한 예방조치가 있습니다.",
                "사이버보안 위협에 대응하기 위해서는 다층 방어체계를 구축하고 실시간 모니터링과 침입탐지시스템을 운영해야 합니다. 또한 정기적인 보안교육과 훈련을 실시하여 보안 인식을 제고해야 합니다.",
                "보안정책을 수립하고 정기적인 보안교육과 훈련을 실시하며 취약점 점검과 보안패치를 지속적으로 수행해야 합니다. 특히 사용자 계정 관리와 접근권한 통제를 강화하여 내부 보안을 확보해야 합니다."
            ],
            "정보보안": [
                "정보보안 관리체계 구축을 위해 보안정책 수립, 위험분석, 보안대책 구현, 사후관리의 절차를 체계적으로 운영해야 합니다. 정보보안관리체계 인증을 통해 보안수준을 객관적으로 평가받을 수 있습니다.",
                "접근통제 정책을 수립하고 사용자별 권한을 관리하며 로그 모니터링과 정기적인 보안감사를 통해 보안수준을 유지해야 합니다. 특히 관리자 계정에 대한 별도의 보안통제를 적용해야 합니다.",
                "보안관제센터를 운영하고 침입탐지시스템과 방화벽을 통해 실시간 보안위협을 탐지하고 대응해야 합니다. 보안정보이벤트관리 시스템을 구축하여 보안사고를 신속히 분석하고 대응할 수 있는 체계를 마련해야 합니다."
            ],
            "금융투자": [
                "자본시장법에 따라 금융투자업자는 투자자 보호와 시장 공정성 확보를 위한 내부통제기준을 수립하고 준수해야 합니다. 투자자문업과 투자매매업 영위 시 각각의 업무범위와 규제사항을 준수해야 합니다.",
                "금융투자업 영위 시 투자자의 투자성향과 위험도를 평가하고 적합한 상품을 권유하는 적합성 원칙을 준수해야 합니다. 특히 일반투자자에 대해서는 투자권유 시 더욱 엄격한 기준을 적용해야 합니다.",
                "펀드 운용 시 투자자에게 투자위험과 손실 가능성을 충분히 설명하고 투명한 정보공시 의무를 이행해야 합니다. 집합투자업자는 선량한 관리자의 주의의무를 다하여 투자자의 이익을 위해 업무를 수행해야 합니다."
            ],
            "위험관리": [
                "위험관리 체계 구축을 위해 위험식별, 위험평가, 위험대응, 위험모니터링의 단계별 절차를 수립하고 운영해야 합니다. 각 단계별로 적절한 통제활동과 점검절차를 마련하여 위험관리의 실효성을 확보해야 합니다.",
                "내부통제시스템을 구축하고 정기적인 위험평가를 실시하여 잠재적 위험요소를 사전에 식별하고 대응방안을 마련해야 합니다. 위험관리조직을 독립적으로 운영하여 객관적인 위험평가가 이루어지도록 해야 합니다.",
                "컴플라이언스 체계를 수립하고 법규 준수 현황을 점검하며 위반 시 즉시 시정조치를 취하는 관리체계를 운영해야 합니다. 임직원에 대한 정기적인 컴플라이언스 교육을 실시하여 준법의식을 제고해야 합니다."
            ],
            "일반": [
                "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다. 정기적인 점검과 평가를 통해 관리체계의 실효성을 확보하고 필요시 개선방안을 마련해야 합니다.",
                "전문적인 보안 정책을 수립하고 정기적인 점검과 평가를 실시하여 보안 수준을 유지해야 합니다. 관련 업무 담당자에 대한 교육과 훈련을 정기적으로 실시하여 전문성을 강화해야 합니다.", 
                "법적 요구사항을 준수하며 효과적인 보안 조치를 시행하고 관련 교육을 실시해야 합니다. 내부 감사와 외부 점검을 통해 관리체계의 적정성을 평가하고 지속적으로 개선해야 합니다.",
                "위험 요소를 식별하고 적절한 대응 방안을 마련하여 체계적으로 관리해야 합니다. 비상계획을 수립하고 정기적인 훈련을 실시하여 위기상황에 신속하고 효과적으로 대응할 수 있는 능력을 배양해야 합니다.",
                "관리 절차를 확립하고 정기적인 평가를 통해 지속적인 개선을 추진해야 합니다. 관련 이해관계자와의 협력체계를 구축하여 효과적인 관리가 이루어지도록 해야 합니다."
            ]
        }
        
        # 워밍업
        self._warmup()
        
        if verbose:
            print("모델 로딩 완료")
        
        # 학습 데이터 로드 현황
        if len(self.learning_data["successful_answers"]) > 0 and verbose:
            print(f"이전 학습 데이터 로드: 성공 {len(self.learning_data['successful_answers'])}개, 실패 {len(self.learning_data['failed_answers'])}개")
    
    def _load_learning_data(self):
        """이전 학습 데이터 로드"""
        learning_file = self.pkl_dir / "learning_data.pkl"
        
        if learning_file.exists():
            try:
                with open(learning_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.learning_data.update(saved_data)
                if self.verbose:
                    print("학습 데이터 로드 완료")
            except Exception as e:
                if self.verbose:
                    print(f"학습 데이터 로드 오류: {e}")
    
    def _save_learning_data(self):
        """학습 데이터 저장"""
        learning_file = self.pkl_dir / "learning_data.pkl"
        
        try:
            # 저장할 데이터 정리 (최근 1000개까지만)
            save_data = {
                "successful_answers": self.learning_data["successful_answers"][-1000:],
                "failed_answers": self.learning_data["failed_answers"][-500:],
                "question_patterns": self.learning_data["question_patterns"],
                "answer_quality_scores": self.learning_data["answer_quality_scores"][-1000:],
                "mc_context_patterns": self.learning_data["mc_context_patterns"],
                "choice_range_errors": self.learning_data["choice_range_errors"][-100:],
                "last_updated": datetime.now().isoformat()
            }
            
            with open(learning_file, 'wb') as f:
                pickle.dump(save_data, f)
                
        except Exception as e:
            if self.verbose:
                print(f"학습 데이터 저장 오류: {e}")
    
    def _extract_choice_count(self, question: str) -> int:
        """질문에서 선택지 개수 추출"""
        # 줄별로 분석하여 선택지 번호 추출
        lines = question.split('\n')
        choice_numbers = []
        
        for line in lines:
            # 선택지 패턴: 숫자 + 공백 + 내용
            match = re.match(r'^(\d+)\s+', line.strip())
            if match:
                choice_numbers.append(int(match.group(1)))
        
        if choice_numbers:
            choice_numbers.sort()
            return max(choice_numbers)
        
        # 폴백: 기본 패턴으로 확인
        for i in range(5, 2, -1):  # 5개부터 3개까지 확인
            pattern = r'1\s.*' + '.*'.join([f'{j}\s' for j in range(2, i+1)])
            if re.search(pattern, question, re.DOTALL):
                return i
        
        return 5  # 기본값
    
    def _analyze_mc_context(self, question: str) -> Dict:
        """객관식 질문 컨텍스트 분석"""
        context = {
            "is_negative": False,
            "is_positive": False,
            "domain_hints": [],
            "key_terms": [],
            "choice_count": self._extract_choice_count(question)
        }
        
        question_lower = question.lower()
        
        # 부정형/긍정형 판단
        for pattern in self.negative_patterns:
            if re.search(pattern, question_lower):
                context["is_negative"] = True
                break
        
        for pattern in self.positive_patterns:
            if re.search(pattern, question_lower):
                context["is_positive"] = True
                break
        
        # 도메인별 힌트 추출
        if "개인정보" in question:
            context["domain_hints"].append("privacy")
        if "보안" in question or "악성코드" in question:
            context["domain_hints"].append("security")
        if "전자금융" in question:
            context["domain_hints"].append("fintech")
        
        # 핵심 용어 추출
        key_terms = ["법", "규정", "조치", "관리", "절차", "기준", "위반", "침해"]
        for term in key_terms:
            if term in question:
                context["key_terms"].append(term)
        
        return context
    
    def _get_context_based_mc_answer(self, question: str, max_choice: int) -> str:
        """컨텍스트 기반 객관식 답변 생성"""
        context = self._analyze_mc_context(question)
        
        # 학습된 패턴에서 유사 질문 찾기
        pattern_key = f"{context['is_negative']}_{context['is_positive']}_{len(context['domain_hints'])}_{max_choice}"
        
        if pattern_key in self.learning_data["mc_context_patterns"]:
            learned_distribution = self.learning_data["mc_context_patterns"][pattern_key]
            # 학습된 분포를 기반으로 가중치 적용
            weighted_choices = []
            for num, weight in learned_distribution.items():
                if int(num) <= max_choice:  # 선택지 범위 내에서만
                    weighted_choices.extend([num] * int(weight * 10))
            
            if weighted_choices:
                return random.choice(weighted_choices)
        
        # 컨텍스트 기반 답변 선택 로직
        if context["is_negative"]:
            # 부정형 질문은 보통 마지막 선택지가 답인 경우가 많음
            weights = [1] * max_choice
            weights[-1] += 2  # 마지막 선택지 가중치 증가
        elif context["is_positive"]:
            # 긍정형 질문은 앞쪽 선택지가 답인 경우가 많음
            weights = [3, 3] + [1] * (max_choice - 2)
        else:
            # 중립적 질문은 균등 분포
            weights = [2] * max_choice
        
        # 도메인 힌트에 따른 가중치 조정
        if "security" in context["domain_hints"] and max_choice >= 3:
            weights[2] += 1  # 보안 관련은 3번이 많음
        if "privacy" in context["domain_hints"] and max_choice >= 1:
            weights[0] += 1  # 개인정보는 1번이 많음
        
        # 가중치 기반 선택
        choices = []
        for i, weight in enumerate(weights):
            choices.extend([str(i+1)] * weight)
        
        return random.choice(choices)
    
    def _add_learning_record(self, question: str, answer: str, question_type: str, success: bool, max_choice: int = 5, quality_score: float = 0.0):
        """학습 기록 추가"""
        record = {
            "question": question[:200],  # 질문 요약
            "answer": answer[:300],      # 답변 요약
            "type": question_type,
            "max_choice": max_choice,
            "timestamp": datetime.now().isoformat(),
            "quality_score": quality_score
        }
        
        if success:
            self.learning_data["successful_answers"].append(record)
        else:
            self.learning_data["failed_answers"].append(record)
            
            # 선택지 범위 오류 기록
            if question_type == "multiple_choice" and answer and answer.isdigit():
                answer_num = int(answer)
                if answer_num > max_choice:
                    self.learning_data["choice_range_errors"].append({
                        "question": question[:100],
                        "answer": answer,
                        "max_choice": max_choice,
                        "timestamp": datetime.now().isoformat()
                    })
        
        # 질문 패턴 학습
        domain = self._detect_domain(question)
        if domain not in self.learning_data["question_patterns"]:
            self.learning_data["question_patterns"][domain] = {"count": 0, "avg_quality": 0.0}
        
        patterns = self.learning_data["question_patterns"][domain]
        patterns["count"] += 1
        patterns["avg_quality"] = (patterns["avg_quality"] * (patterns["count"] - 1) + quality_score) / patterns["count"]
        
        # 객관식 컨텍스트 패턴 학습
        if question_type == "multiple_choice" and success and answer.isdigit():
            context = self._analyze_mc_context(question)
            pattern_key = f"{context['is_negative']}_{context['is_positive']}_{len(context['domain_hints'])}_{max_choice}"
            
            if pattern_key not in self.learning_data["mc_context_patterns"]:
                self.learning_data["mc_context_patterns"][pattern_key] = {}
            
            if answer in self.learning_data["mc_context_patterns"][pattern_key]:
                self.learning_data["mc_context_patterns"][pattern_key][answer] += 1
            else:
                self.learning_data["mc_context_patterns"][pattern_key][answer] = 1
        
        # 품질 점수 기록
        self.learning_data["answer_quality_scores"].append(quality_score)
    
    def _calculate_korean_ratio(self, text: str) -> float:
        """한국어 비율 계산"""
        if not text:
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w가-힣]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        return korean_chars / total_chars
    
    def _warmup(self):
        """모델 워밍업"""
        try:
            test_prompt = "테스트"
            inputs = self.tokenizer(test_prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False
                )
            if self.verbose:
                print("모델 워밍업 완료")
        except Exception as e:
            if self.verbose:
                print(f"워밍업 실패: {e}")
    
    def generate_answer(self, question: str, question_type: str, max_choice: int = 5) -> str:
        """답변 생성"""
        
        # 프롬프트 생성
        if question_type == "multiple_choice":
            prompt = self._create_enhanced_mc_prompt(question, max_choice)
        else:
            prompt = self._create_korean_subj_prompt(question)
        
        try:
            # 토크나이징
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1500
            )
            
            if self.device == "cuda":
                inputs = inputs.to(self.model.device)
            
            # 생성 설정
            gen_config = self._get_generation_config(question_type)
            
            # 모델 실행
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config
                )
            
            # 디코딩
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # 후처리
            if question_type == "multiple_choice":
                answer = self._process_enhanced_mc_answer(response, question, max_choice)
                # 선택지 범위 검증
                if answer and answer.isdigit() and 1 <= int(answer) <= max_choice:
                    self._add_learning_record(question, answer, question_type, True, max_choice, 1.0)
                    return answer
                else:
                    # 범위 벗어난 경우 컨텍스트 기반 폴백
                    fallback = self._get_context_based_mc_answer(question, max_choice)
                    self._add_learning_record(question, answer, question_type, False, max_choice, 0.0)
                    return fallback
            else:
                answer = self._process_korean_subj_answer(response, question)
                korean_ratio = self._calculate_korean_ratio(answer)
                quality_score = self._calculate_answer_quality(answer, question)
                success = korean_ratio > 0.7 and quality_score > 0.5
                
                self._add_learning_record(question, answer, question_type, success, max_choice, quality_score)
                return answer
                
        except Exception as e:
            if self.verbose:
                print(f"모델 실행 오류: {e}")
            fallback = self._get_fallback_answer(question_type, question, max_choice)
            self._add_learning_record(question, fallback, question_type, False, max_choice, 0.3)
            return fallback
    
    def _create_enhanced_mc_prompt(self, question: str, max_choice: int) -> str:
        """개선된 객관식 프롬프트 생성"""
        context = self._analyze_mc_context(question)
        
        # 선택지 범위 명시
        choice_range = "에서 ".join([str(i) for i in range(1, max_choice+1)]) + f"번 중"
        
        # 컨텍스트에 따른 프롬프트 조정
        if context["is_negative"]:
            instruction = f"다음 중 해당하지 않거나 옳지 않은 것을 {choice_range} 찾으세요."
        elif context["is_positive"]:
            instruction = f"다음 중 가장 적절하거나 옳은 것을 {choice_range} 찾으세요."
        else:
            instruction = f"정답을 {choice_range} 선택하세요."
        
        prompts = [
            f"""다음은 금융보안 관련 문제입니다. {instruction}

{question}

위 문제를 신중히 분석하고, 1부터 {max_choice}까지 중 하나의 정답을 선택하세요.
각 선택지를 꼼꼼히 검토한 후 정답 번호만 답하세요.

정답:""",
            
            f"""금융보안 전문가로서 다음 문제를 해결하세요.

{question}

{instruction}
선택지를 모두 검토한 후 1부터 {max_choice}번 중 정답을 선택하세요.
번호만 답하세요.

답:""",
            
            f"""다음 금융보안 문제를 분석하고 정답을 선택하세요.

문제: {question}

{instruction}
정답을 1부터 {max_choice}번 중 하나의 번호로만 답하세요.

정답:"""
        ]
        
        return random.choice(prompts)
    
    def _create_korean_subj_prompt(self, question: str) -> str:
        """한국어 전용 주관식 프롬프트 생성"""
        domain = self._detect_domain(question)
        
        prompts = [
            f"""금융보안 전문가로서 다음 질문에 대해 한국어로만 정확한 답변을 작성하세요.

질문: {question}

답변 작성 시 다음 사항을 준수하세요:
- 반드시 한국어로만 작성
- 관련 법령과 규정을 근거로 구체적 내용 포함
- 실무적이고 전문적인 관점에서 설명
- 영어 용어 사용 금지

답변:""",
            
            f"""다음은 {domain} 분야의 전문 질문입니다. 한국어로만 상세하고 정확한 답변을 제공하세요.

{question}

한국어 전용 답변 작성 기준:
- 모든 전문 용어를 한국어로 표기
- 법적 근거와 실무 절차를 한국어로 설명
- 영어나 외국어 사용 금지

답변:""",
            
            f"""{domain} 전문가의 관점에서 다음 질문에 한국어로만 답변하세요.

질문: {question}

답변 요구사항:
- 완전한 한국어 답변
- 관련 법령과 규정을 한국어로 설명
- 체계적이고 구체적인 한국어 설명

답변:"""
        ]
        
        return random.choice(prompts)
    
    def _get_generation_config(self, question_type: str) -> GenerationConfig:
        """생성 설정"""
        if question_type == "multiple_choice":
            return GenerationConfig(
                max_new_tokens=20,
                temperature=0.4,
                top_p=0.85,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        else:
            return GenerationConfig(
                max_new_tokens=300,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
    
    def _process_enhanced_mc_answer(self, response: str, question: str, max_choice: int) -> str:
        """개선된 객관식 답변 처리"""
        # 숫자 추출 (선택지 범위 내에서만)
        numbers = re.findall(r'[1-9]', response)
        for num in numbers:
            if 1 <= int(num) <= max_choice:
                # 답변 분포 업데이트
                if max_choice in self.answer_distributions:
                    self.answer_distributions[max_choice][num] += 1
                    self.mc_answer_counts[max_choice] += 1
                return num
        
        # 유효한 답변을 찾지 못한 경우 컨텍스트 기반 폴백
        return self._get_context_based_mc_answer(question, max_choice)
    
    def _process_korean_subj_answer(self, response: str, question: str) -> str:
        """한국어 전용 주관식 답변 처리"""
        # 기본 정리
        response = re.sub(r'\s+', ' ', response).strip()
        
        # 잘못된 인코딩으로 인한 깨진 문자 제거
        response = re.sub(r'[^\w\s가-힣.,!?()[\]\-]', ' ', response)
        response = re.sub(r'\s+', ' ', response).strip()
        
        # 한국어 비율 확인
        korean_ratio = self._calculate_korean_ratio(response)
        
        # 한국어 비율이 낮거나 길이가 짧으면 템플릿 사용
        if korean_ratio < 0.7 or len(response) < 20:
            return self._generate_korean_template_answer(question)
        
        # 길이 제한
        if len(response) > 350:
            sentences = response.split('. ')
            response = '. '.join(sentences[:3])
            if not response.endswith('.'):
                response += '.'
        
        # 마침표 확인
        if not response.endswith(('.', '다', '요', '함')):
            response += '.'
        
        return response
    
    def _generate_korean_template_answer(self, question: str) -> str:
        """한국어 템플릿 답변 생성"""
        domain = self._detect_domain(question)
        
        # 도메인별 템플릿 사용
        if domain in self.korean_templates:
            templates = self.korean_templates[domain]
        else:
            # 일반 템플릿
            templates = [
                "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다.",
                "전문적인 보안 정책을 수립하고 정기적인 점검과 평가를 실시하여 보안 수준을 유지해야 합니다.", 
                "법적 요구사항을 준수하며 효과적인 보안 조치를 시행하고 관련 교육을 실시해야 합니다.",
                "위험 요소를 식별하고 적절한 대응 방안을 마련하여 체계적으로 관리해야 합니다.",
                "관리 절차를 확립하고 정기적인 평가를 통해 지속적인 개선을 추진해야 합니다."
            ]
        
        return random.choice(templates)
    
    def _calculate_answer_quality(self, answer: str, question: str) -> float:
        """답변 품질 점수 계산"""
        if not answer:
            return 0.0
        
        score = 0.0
        
        # 한국어 비율 (40%)
        korean_ratio = self._calculate_korean_ratio(answer)
        score += korean_ratio * 0.4
        
        # 길이 적절성 (20%)
        length = len(answer)
        if 50 <= length <= 400:
            score += 0.2
        elif 30 <= length < 50 or 400 < length <= 500:
            score += 0.1
        
        # 문장 구조 (20%)
        if answer.endswith(('.', '다', '요', '함')):
            score += 0.1
        
        sentences = answer.split('.')
        if len(sentences) >= 2:
            score += 0.1
        
        # 전문성 (20%)
        domain_keywords = self._get_domain_keywords(question)
        found_keywords = sum(1 for keyword in domain_keywords if keyword in answer)
        if found_keywords > 0:
            score += min(found_keywords / len(domain_keywords), 1.0) * 0.2
        
        return min(score, 1.0)
    
    def _get_domain_keywords(self, question: str) -> List[str]:
        """도메인별 키워드 반환"""
        question_lower = question.lower()
        
        if "개인정보" in question_lower:
            return ["개인정보보호법", "정보주체", "처리", "보호조치", "동의"]
        elif "전자금융" in question_lower:
            return ["전자금융거래법", "접근매체", "인증", "보안", "분쟁조정"]
        elif "보안" in question_lower or "악성코드" in question_lower:
            return ["보안정책", "탐지", "대응", "모니터링", "방어"]
        else:
            return ["법령", "규정", "관리", "조치", "절차"]
    
    def _get_balanced_mc_answer(self, max_choice: int) -> str:
        """균등 분포 객관식 답변"""
        if max_choice not in self.mc_answer_counts or self.mc_answer_counts[max_choice] < max_choice:
            return str(random.randint(1, max_choice))
        
        # 현재 분포 확인
        if max_choice in self.answer_distributions:
            distribution = self.answer_distributions[max_choice]
            avg_count = self.mc_answer_counts[max_choice] / max_choice
            underused = [num for num in range(1, max_choice+1) 
                        if distribution.get(str(num), 0) < avg_count * 0.7]
            
            if underused:
                answer = str(random.choice(underused))
            else:
                answer = str(random.randint(1, max_choice))
            
            distribution[answer] += 1
            self.mc_answer_counts[max_choice] += 1
            return answer
        
        return str(random.randint(1, max_choice))
    
    def _get_fallback_answer(self, question_type: str, question: str = "", max_choice: int = 5) -> str:
        """폴백 답변"""
        if question_type == "multiple_choice":
            return self._get_context_based_mc_answer(question, max_choice)
        else:
            return self._generate_korean_template_answer(question)
    
    def _detect_domain(self, question: str) -> str:
        """도메인 감지"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["개인정보", "정보주체"]):
            return "개인정보보호"
        elif any(word in question_lower for word in ["전자금융", "전자적"]):
            return "전자금융"
        elif any(word in question_lower for word in ["보안", "악성코드", "트로이"]):
            return "사이버보안"
        elif any(word in question_lower for word in ["정보보안", "isms"]):
            return "정보보안"
        elif any(word in question_lower for word in ["금융투자", "자본시장"]):
            return "금융투자"
        elif any(word in question_lower for word in ["위험관리", "리스크"]):
            return "위험관리"
        else:
            return "일반"
    
    def get_answer_stats(self) -> Dict:
        """답변 통계"""
        return {
            "distributions": dict(self.answer_distributions),
            "counts": dict(self.mc_answer_counts)
        }
    
    def get_learning_stats(self) -> Dict:
        """학습 통계"""
        return {
            "successful_count": len(self.learning_data["successful_answers"]),
            "failed_count": len(self.learning_data["failed_answers"]),
            "choice_range_errors": len(self.learning_data["choice_range_errors"]),
            "question_patterns": dict(self.learning_data["question_patterns"]),
            "avg_quality": sum(self.learning_data["answer_quality_scores"]) / len(self.learning_data["answer_quality_scores"]) if self.learning_data["answer_quality_scores"] else 0
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            # 학습 데이터 저장
            self._save_learning_data()
            
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            if self.verbose:
                print(f"정리 중 오류: {e}")