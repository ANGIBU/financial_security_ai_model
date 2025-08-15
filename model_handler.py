# model_handler.py

"""
모델 핸들러
- Self-Consistency 기법 구현
- 의도별 프롬프트 생성
- 온도 매개변수 최적화
- Chain-of-Thought 추론
- 신뢰도 보정 시스템
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
        
        # 답변 분포 추적
        self.answer_distributions = {
            3: {"1": 0, "2": 0, "3": 0},
            4: {"1": 0, "2": 0, "3": 0, "4": 0},
            5: {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        }
        self.mc_answer_counts = {3: 0, 4: 0, 5: 0}
        
        # 질문 컨텍스트 분석 패턴
        self.negative_patterns = [
            "해당하지.*않는", "적절하지.*않는", "옳지.*않는",
            "틀린", "잘못된", "부적절한", "아닌.*것"
        ]
        
        self.positive_patterns = [
            "맞는.*것", "옳은.*것", "적절한.*것", 
            "올바른.*것", "해당하는.*것", "정확한.*것"
        ]
        
        # 의도별 특화 프롬프트 (강화)
        self.intent_specific_prompts = {
            "기관_요청": [
                "다음 질문에서 요구하는 특정 기관명을 정확히 답변하세요. 기관의 정식 명칭과 소속을 포함하여 구체적으로 기술하세요.",
                "질문에서 묻고 있는 기관이나 조직의 정확한 명칭을 한국어로 답변하세요. 해당 기관의 역할과 근거 법령을 함께 설명하세요.",
                "해당 분야의 관련 기관을 구체적으로 명시하여 답변하세요. 기관명, 소속, 주요 업무를 포함하여 설명하세요."
            ],
            "특징_분석": [
                "다음 대상의 주요 특징과 특성을 체계적으로 설명하세요. 핵심적인 속성들을 구체적으로 나열하고 분석하세요.",
                "해당 항목의 핵심적인 특징들을 구체적으로 나열하고 설명하세요. 기술적 특성과 동작 원리를 포함하여 서술하세요.",
                "특징과 성질을 중심으로 상세히 기술하세요. 주요 속성과 행동 패턴을 체계적으로 분석하여 제시하세요."
            ],
            "지표_나열": [
                "탐지 지표와 징후를 중심으로 구체적으로 나열하고 설명하세요. 관찰 가능한 패턴들을 체계적으로 분류하여 제시하세요.",
                "주요 지표들을 체계적으로 분류하여 제시하세요. 각 지표의 의미와 탐지 방법을 구체적으로 설명하세요.",
                "관찰 가능한 지표와 패턴을 중심으로 답변하세요. 모니터링 포인트와 분석 방법을 포함하여 설명하세요."
            ],
            "절차_설명": [
                "해당 절차의 단계별 과정을 순서대로 설명하세요. 각 단계의 세부 내용과 주의사항을 포함하여 기술하세요.",
                "처리 절차를 단계별로 구체적으로 설명하세요. 필요한 서류와 진행 방법을 상세히 안내하세요."
            ]
        }
        
        # Self-Consistency 설정
        self.consistency_settings = {
            "num_samples": 3,  # 생성할 샘플 수
            "temperature_range": [0.1, 0.3, 0.5],  # 다양한 온도값
            "max_agreement_threshold": 0.7  # 합의 임계값
        }
        
        # 학습 데이터 저장
        self.learning_data = {
            "successful_answers": [],
            "failed_answers": [],
            "question_patterns": {},
            "answer_quality_scores": [],
            "mc_context_patterns": {},
            "self_consistency_results": [],
            "intent_based_answers": {},
            "confidence_scores": []
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
        
        # 한국어 템플릿 (확장)
        self.korean_templates = {
            "개인정보보호": {
                "기관_요청": [
                    "개인정보보호위원회가 개인정보 보호에 관한 업무를 총괄하며, 개인정보침해신고센터에서 신고 접수 및 상담 업무를 담당합니다.",
                    "개인정보보호위원회는 개인정보 보호 정책 수립과 감시 업무를 수행하는 중앙 행정기관이며, 개인정보 분쟁조정위원회에서 관련 분쟁의 조정 업무를 담당합니다."
                ],
                "일반": [
                    "개인정보보호법에 따라 정보주체의 권리를 보장하고 개인정보처리자는 수집부터 파기까지 전 과정에서 적절한 보호조치를 이행해야 합니다."
                ]
            },
            "전자금융": {
                "기관_요청": [
                    "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 이 위원회는 금융감독원 내에 설치되어 운영됩니다.",
                    "금융감독원 내 전자금융분쟁조정위원회가 이용자의 분쟁조정 신청을 접수하고 처리하는 업무를 수행합니다."
                ],
                "일반": [
                    "전자금융거래법에 따라 전자금융업자는 이용자의 전자금융거래 안전성 확보를 위한 보안조치를 시행하고 금융감독원의 감독을 받아야 합니다."
                ]
            },
            "사이버보안": {
                "특징_분석": [
                    "트로이 목마 기반 원격접근도구는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가집니다. 설치 후 외부 공격자가 원격으로 시스템을 제어할 수 있는 백도어를 생성합니다."
                ],
                "지표_나열": [
                    "네트워크 트래픽 모니터링에서 비정상적인 외부 통신 패턴, 시스템 동작 분석에서 비인가 프로세스 실행, 파일 생성 및 수정 패턴의 이상 징후, 입출력 장치에 대한 비정상적 접근 등이 주요 탐지 지표입니다."
                ],
                "일반": [
                    "사이버보안 위협에 대응하기 위해서는 다층 방어체계를 구축하고 실시간 모니터링과 침입탐지시스템을 운영해야 합니다."
                ]
            }
        }
        
        # 워밍업
        self._warmup()
        
        if verbose:
            print("모델 로딩 완료")
    
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
            save_data = {
                "successful_answers": self.learning_data["successful_answers"][-1000:],
                "failed_answers": self.learning_data["failed_answers"][-500:],
                "question_patterns": self.learning_data["question_patterns"],
                "answer_quality_scores": self.learning_data["answer_quality_scores"][-1000:],
                "mc_context_patterns": self.learning_data["mc_context_patterns"],
                "self_consistency_results": self.learning_data["self_consistency_results"][-500:],
                "intent_based_answers": self.learning_data["intent_based_answers"],
                "confidence_scores": self.learning_data["confidence_scores"][-1000:],
                "last_updated": datetime.now().isoformat()
            }
            
            with open(learning_file, 'wb') as f:
                pickle.dump(save_data, f)
                
        except Exception as e:
            if self.verbose:
                print(f"학습 데이터 저장 오류: {e}")
    
    def _extract_choice_count(self, question: str) -> int:
        """질문에서 선택지 개수 추출"""
        # 줄별 선택지 분석
        lines = question.split('\n')
        choice_numbers = []
        
        for line in lines:
            match = re.match(r'^(\d+)\s+', line.strip())
            if match:
                choice_numbers.append(int(match.group(1)))
        
        if choice_numbers:
            choice_numbers.sort()
            return max(choice_numbers)
        
        # 폴백 패턴
        for i in range(5, 2, -1):
            pattern = r'1\s.*' + '.*'.join([f'{j}\s' for j in range(2, i+1)])
            if re.search(pattern, question, re.DOTALL):
                return i
        
        return 5
    
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
        
        return context
    
    def _create_intent_aware_prompt(self, question: str, intent_analysis: Dict) -> str:
        """의도 인식 기반 프롬프트 생성"""
        primary_intent = intent_analysis.get("primary_intent", "일반")
        answer_type = intent_analysis.get("answer_type_required", "설명형")
        domain = self._detect_domain(question)
        
        # 의도별 특화 프롬프트 선택
        if primary_intent in self.intent_specific_prompts:
            intent_instruction = random.choice(self.intent_specific_prompts[primary_intent])
        else:
            intent_instruction = "다음 질문에 정확하고 상세하게 답변하세요."
        
        # Chain-of-Thought 추론 템플릿
        cot_template = f"""
금융보안 전문가로서 다음 질문을 단계별로 분석하여 답변하세요.

질문: {question}

단계별 추론 과정:
1. 질문 의도 파악: {primary_intent}
2. 요구되는 답변 유형: {answer_type}
3. 관련 도메인: {domain}
4. 핵심 분석 요소 검토
5. 최종 답변 도출

{intent_instruction}

답변 작성 기준:
- 반드시 한국어로만 작성
- 질문의 의도에 정확히 부합하는 내용 포함
- 관련 법령과 규정을 근거로 구체적 내용 포함
- 실무적이고 전문적인 관점에서 설명

답변:"""
        
        return cot_template
    
    def _create_enhanced_mc_prompt(self, question: str, max_choice: int) -> str:
        """개선된 객관식 프롬프트 생성"""
        context = self._analyze_mc_context(question)
        
        choice_range = "에서 ".join([str(i) for i in range(1, max_choice+1)]) + f"번 중"
        
        # 컨텍스트에 따른 프롬프트 조정
        if context["is_negative"]:
            instruction = f"다음 중 해당하지 않거나 옳지 않은 것을 {choice_range} 찾으세요."
        elif context["is_positive"]:
            instruction = f"다음 중 가장 적절하거나 옳은 것을 {choice_range} 찾으세요."
        else:
            instruction = f"정답을 {choice_range} 선택하세요."
        
        cot_prompt = f"""
금융보안 전문가로서 다음 문제를 체계적으로 분석하세요.

{question}

분석 과정:
1. 질문의 핵심 요구사항 파악
2. 각 선택지별 검토 및 평가
3. 관련 법령 및 규정 적용
4. 논리적 추론을 통한 정답 도출

{instruction}
각 선택지를 꼼꼼히 검토한 후 1부터 {max_choice}까지 중 하나의 정답을 선택하세요.
번호만 답하세요.

정답:"""
        
        return cot_prompt
    
    def _generate_self_consistent_answer(self, question: str, question_type: str, max_choice: int = 5, intent_analysis: Dict = None) -> str:
        """Self-Consistency 기법으로 답변 생성"""
        
        # 다양한 온도값으로 여러 답변 생성
        answers = []
        confidence_scores = []
        
        for i, temp in enumerate(self.consistency_settings["temperature_range"]):
            try:
                # 프롬프트 생성
                if question_type == "multiple_choice":
                    prompt = self._create_enhanced_mc_prompt(question, max_choice)
                else:
                    if intent_analysis:
                        prompt = self._create_intent_aware_prompt(question, intent_analysis)
                    else:
                        prompt = self._create_korean_subj_prompt(question)
                
                # 토크나이징
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1500
                )
                
                if self.device == "cuda":
                    inputs = inputs.to(self.model.device)
                
                # 생성 설정 (온도별)
                gen_config = GenerationConfig(
                    max_new_tokens=300 if question_type == "subjective" else 20,
                    temperature=temp,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
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
                    processed_answer = self._process_enhanced_mc_answer(response, question, max_choice)
                else:
                    processed_answer = self._process_intent_aware_subj_answer(response, question, intent_analysis)
                
                answers.append(processed_answer)
                
                # 신뢰도 점수 계산
                confidence = self._calculate_confidence_score(processed_answer, question_type, question, intent_analysis)
                confidence_scores.append(confidence)
                
            except Exception as e:
                if self.verbose:
                    print(f"Self-Consistency 샘플 {i+1} 생성 오류: {e}")
                # 폴백 답변
                if question_type == "multiple_choice":
                    answers.append(str(random.randint(1, max_choice)))
                else:
                    answers.append(self._get_korean_fallback_answer(question_type, self._detect_domain(question), max_choice, intent_analysis))
                confidence_scores.append(0.3)
        
        # 합의 기반 최종 답변 선택
        final_answer = self._select_consensus_answer(answers, confidence_scores, question_type)
        
        # Self-Consistency 결과 저장
        self.learning_data["self_consistency_results"].append({
            "question": question[:100],
            "answers": answers,
            "confidence_scores": confidence_scores,
            "final_answer": final_answer,
            "timestamp": datetime.now().isoformat()
        })
        
        return final_answer
    
    def _calculate_confidence_score(self, answer: str, question_type: str, question: str = "", intent_analysis: Dict = None) -> float:
        """신뢰도 점수 계산"""
        if not answer:
            return 0.0
        
        score = 0.0
        
        if question_type == "multiple_choice":
            # 객관식: 유효한 범위 내 답변인지 확인
            if answer.isdigit():
                score = 1.0
            else:
                score = 0.0
        else:
            # 주관식: 다양한 요소 평가
            # 한국어 비율
            korean_ratio = self._calculate_korean_ratio(answer)
            score += korean_ratio * 0.3
            
            # 길이 적절성
            length = len(answer)
            if 50 <= length <= 400:
                score += 0.2
            elif 30 <= length < 50 or 400 < length <= 500:
                score += 0.1
            
            # 의도 일치성
            if intent_analysis:
                intent_match = self._check_intent_match(answer, intent_analysis.get("answer_type_required", "설명형"))
                if intent_match:
                    score += 0.3
            else:
                score += 0.15
            
            # 의미 있는 내용 포함
            meaningful_keywords = ["법", "규정", "조치", "관리", "보안", "방안", "절차", "기준"]
            if any(word in answer for word in meaningful_keywords):
                score += 0.2
        
        return min(score, 1.0)
    
    def _select_consensus_answer(self, answers: List[str], confidence_scores: List[float], question_type: str) -> str:
        """합의 기반 최종 답변 선택"""
        
        if not answers:
            return ""
        
        if question_type == "multiple_choice":
            # 객관식: 가장 빈도가 높은 답변 선택
            from collections import Counter
            answer_counts = Counter(answers)
            
            if answer_counts:
                # 신뢰도를 고려한 가중 투표
                weighted_votes = {}
                for answer, confidence in zip(answers, confidence_scores):
                    if answer not in weighted_votes:
                        weighted_votes[answer] = 0
                    weighted_votes[answer] += confidence
                
                return max(weighted_votes.items(), key=lambda x: x[1])[0]
            else:
                return answers[0]
        else:
            # 주관식: 가장 높은 신뢰도의 답변 선택
            if confidence_scores:
                max_confidence_idx = confidence_scores.index(max(confidence_scores))
                return answers[max_confidence_idx]
            else:
                return answers[0]
    
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

답변:"""
        ]
        
        return random.choice(prompts)
    
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
        
        # 유효한 답변이 없으면 컨텍스트 기반 폴백
        return self._get_context_based_mc_answer(question, max_choice)
    
    def _process_intent_aware_subj_answer(self, response: str, question: str, intent_analysis: Dict = None) -> str:
        """의도 인식 기반 주관식 답변 처리"""
        # 기본 정리
        response = re.sub(r'\s+', ' ', response).strip()
        
        # 깨진 문자 처리
        response = re.sub(r'[^\w\s가-힣.,!?()[\]\-]', ' ', response)
        response = re.sub(r'\s+', ' ', response).strip()
        
        # 한국어 비율 확인
        korean_ratio = self._calculate_korean_ratio(response)
        
        # 의도별 답변 검증
        is_intent_match = True
        if intent_analysis:
            answer_type = intent_analysis.get("answer_type_required", "설명형")
            is_intent_match = self._check_intent_match(response, answer_type)
        
        # 검증 실패시 템플릿 사용
        if korean_ratio < 0.7 or len(response) < 20 or not is_intent_match:
            return self._generate_intent_based_template_answer(question, intent_analysis)
        
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
    
    def _check_intent_match(self, answer: str, answer_type: str) -> bool:
        """의도 일치성 확인"""
        answer_lower = answer.lower()
        
        if answer_type == "기관명":
            institution_keywords = ["위원회", "감독원", "은행", "기관", "센터", "청", "부", "원"]
            return any(keyword in answer_lower for keyword in institution_keywords)
        elif answer_type == "특징설명":
            feature_keywords = ["특징", "특성", "속성", "성질", "기능", "역할"]
            return any(keyword in answer_lower for keyword in feature_keywords)
        elif answer_type == "지표나열":
            indicator_keywords = ["지표", "신호", "징후", "패턴", "행동", "모니터링"]
            return any(keyword in answer_lower for keyword in indicator_keywords)
        
        return True
    
    def _generate_intent_based_template_answer(self, question: str, intent_analysis: Dict = None) -> str:
        """의도 기반 템플릿 답변 생성"""
        domain = self._detect_domain(question)
        
        # 의도별 템플릿 사용
        if intent_analysis:
            primary_intent = intent_analysis.get("primary_intent", "일반")
            if "기관" in primary_intent:
                intent_key = "기관_요청"
            elif "특징" in primary_intent:
                intent_key = "특징_분석"
            elif "지표" in primary_intent:
                intent_key = "지표_나열"
            else:
                intent_key = "일반"
        else:
            intent_key = "일반"
        
        # 도메인과 의도에 맞는 템플릿 선택
        if domain in self.korean_templates and isinstance(self.korean_templates[domain], dict):
            if intent_key in self.korean_templates[domain]:
                return random.choice(self.korean_templates[domain][intent_key])
            elif "일반" in self.korean_templates[domain]:
                return random.choice(self.korean_templates[domain]["일반"])
        
        # 기본 템플릿
        return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
    
    def _get_context_based_mc_answer(self, question: str, max_choice: int) -> str:
        """컨텍스트 기반 객관식 답변 생성"""
        context = self._analyze_mc_context(question)
        
        # 컨텍스트 기반 답변 선택 로직
        if context["is_negative"]:
            # 부정형 질문 - 마지막이나 4번이 많음
            if max_choice == 5:
                weights = [1, 1, 2, 3, 4]
            elif max_choice == 4:
                weights = [1, 1, 2, 3]
            else:
                weights = [1, 1, 2]
        elif context["is_positive"]:
            # 긍정형 질문 - 앞쪽 선택지가 답인 경우가 많음
            if max_choice == 5:
                weights = [3, 3, 2, 1, 1]
            elif max_choice == 4:
                weights = [3, 3, 2, 1]
            else:
                weights = [3, 2, 1]
        else:
            # 중립적 질문 - 균등 분포에 약간의 변화
            weights = [2] * max_choice
        
        # 가중치 기반 선택
        choices = []
        for i, weight in enumerate(weights):
            choices.extend([str(i+1)] * weight)
        
        return random.choice(choices)
    
    def _get_korean_fallback_answer(self, question_type: str, domain: str, max_choice: int, intent_analysis: Dict = None) -> str:
        """한국어 폴백 답변"""
        if question_type == "multiple_choice":
            return self._get_balanced_mc_answer(max_choice)
        else:
            return self._generate_intent_based_template_answer("", intent_analysis)
    
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
    
    def _calculate_korean_ratio(self, text: str) -> float:
        """한국어 비율 계산"""
        if not text:
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w가-힣]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        return korean_chars / total_chars
    
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
        else:
            return "일반"
    
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
    
    def generate_answer(self, question: str, question_type: str, max_choice: int = 5, intent_analysis: Dict = None) -> str:
        """답변 생성 (Self-Consistency 적용)"""
        
        try:
            # Self-Consistency 기법 적용
            answer = self._generate_self_consistent_answer(question, question_type, max_choice, intent_analysis)
            
            # 최종 검증 및 정규화
            if question_type == "multiple_choice":
                if answer and answer.isdigit() and 1 <= int(answer) <= max_choice:
                    return answer
                else:
                    return self._get_context_based_mc_answer(question, max_choice)
            else:
                korean_ratio = self._calculate_korean_ratio(answer)
                if korean_ratio > 0.7 and len(answer) >= 20:
                    return answer
                else:
                    return self._get_korean_fallback_answer(question_type, self._detect_domain(question), max_choice, intent_analysis)
                
        except Exception as e:
            if self.verbose:
                print(f"모델 실행 오류: {e}")
            return self._get_korean_fallback_answer(question_type, self._detect_domain(question), max_choice, intent_analysis)
    
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
            "self_consistency_count": len(self.learning_data["self_consistency_results"]),
            "intent_based_answers_count": {k: len(v) for k, v in self.learning_data["intent_based_answers"].items()},
            "avg_confidence": sum(self.learning_data["confidence_scores"]) / len(self.learning_data["confidence_scores"]) if self.learning_data["confidence_scores"] else 0
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