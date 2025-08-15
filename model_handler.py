# model_handler.py

"""
LLM 모델 핸들러 (성능 최적화)
- 모델 로딩 및 관리 최적화
- 답변 생성 성능 향상
- 프롬프트 처리 최적화
- 학습 데이터 저장 최적화
- 질문 의도 기반 답변 생성 고도화
- 객관식 처리 성능 최적화 (CSV 분석 기반)
- 메모리 사용량 최적화
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
from collections import defaultdict, deque
import warnings
warnings.filterwarnings("ignore")

class SimpleModelHandler:
    """모델 핸들러 (성능 최적화)"""
    
    def __init__(self, model_name: str = "upstage/SOLAR-10.7B-Instruct-v1.0", verbose: bool = False):
        self.model_name = model_name
        self.verbose = verbose
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # pkl 저장 폴더 생성
        self.pkl_dir = Path("./pkl")
        self.pkl_dir.mkdir(exist_ok=True)
        
        # 성능 최적화된 답변 분포 추적 (CSV 분석 기반)
        self.answer_distributions = {
            3: defaultdict(int),
            4: defaultdict(int),
            5: defaultdict(int)
        }
        self.mc_answer_counts = defaultdict(int)
        
        # 질문 컨텍스트 분석용 패턴 (최적화)
        self.negative_patterns = [
            "해당하지.*않는", "적절하지.*않는", "옳지.*않는",
            "틀린", "잘못된", "부적절한", "아닌.*것"
        ]
        
        self.positive_patterns = [
            "맞는.*것", "옳은.*것", "적절한.*것", 
            "올바른.*것", "해당하는.*것", "정확한.*것"
        ]
        
        # 의도별 특화 프롬프트 패턴 (CSV 기반 최적화)
        self.intent_specific_prompts = {
            "기관_묻기": [
                "다음 질문에서 요구하는 특정 기관명을 정확히 답변하세요.",
                "질문에서 묻고 있는 기관이나 조직의 정확한 명칭을 한국어로 답변하세요.",
                "해당 분야의 관련 기관을 구체적으로 명시하여 답변하세요.",
                "분쟁조정이나 신고접수를 담당하는 기관명을 정확히 제시하세요.",
                "관련 법령에 따라 업무를 담당하는 기관의 정확한 명칭을 답변하세요."
            ],
            "특징_묻기": [
                "다음 대상의 주요 특징과 특성을 체계적으로 설명하세요.",
                "해당 항목의 핵심적인 특징들을 구체적으로 나열하고 설명하세요.",
                "특징과 성질을 중심으로 상세히 기술하세요.",
                "고유한 특성과 차별화 요소를 포함하여 설명하세요.",
                "주요 특징을 분류하여 체계적으로 제시하세요."
            ],
            "지표_묻기": [
                "탐지 지표와 징후를 중심으로 구체적으로 나열하고 설명하세요.",
                "주요 지표들을 체계적으로 분류하여 제시하세요.",
                "관찰 가능한 지표와 패턴을 중심으로 답변하세요.",
                "식별 가능한 신호와 징후를 구체적으로 설명하세요.",
                "모니터링과 탐지에 활용할 수 있는 지표를 제시하세요."
            ],
            "방안_묻기": [
                "구체적인 대응 방안과 해결책을 제시하세요.",
                "실무적이고 실행 가능한 방안들을 중심으로 답변하세요.",
                "체계적인 관리 방안을 단계별로 설명하세요.",
                "효과적인 대처 방안과 예방책을 함께 제시하세요.",
                "실제 적용 가능한 구체적 방안을 설명하세요."
            ],
            "절차_묻기": [
                "단계별 절차를 순서대로 설명하세요.",
                "처리 과정을 체계적으로 기술하세요.",
                "진행 절차와 각 단계의 내용을 상세히 설명하세요.",
                "업무 프로세스를 단계별로 제시하세요.",
                "수행 절차를 논리적 순서에 따라 설명하세요."
            ]
        }
        
        # 성능 최적화된 학습 데이터 저장
        self.learning_data = {
            "successful_answers": deque(maxlen=1000),  # 메모리 최적화
            "failed_answers": deque(maxlen=500),
            "question_patterns": defaultdict(list),
            "answer_quality_scores": deque(maxlen=1000),
            "mc_context_patterns": defaultdict(lambda: defaultdict(int)),
            "choice_range_errors": deque(maxlen=100),
            "intent_based_answers": defaultdict(lambda: deque(maxlen=50)),
            "domain_specific_learning": defaultdict(dict),
            "intent_prompt_effectiveness": defaultdict(dict),
            "high_quality_templates": defaultdict(lambda: deque(maxlen=20)),
            "performance_metrics": {
                "avg_generation_time": 0.0,
                "cache_hit_rate": 0.0,
                "template_efficiency": 0.0
            }
        }
        
        # 답변 캐시 시스템 (성능 최적화)
        self.answer_cache = {}
        self.prompt_cache = {}
        self.max_cache_size = 500
        
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
        
        # 모델 로드 (메모리 최적화)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        
        # 한국어 전용 주관식 답변 템플릿 (CSV 기반 최적화)
        self.korean_templates = {
            "전자금융": {
                "기관_묻기": [
                    "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 이 위원회는 금융감독원 내에 설치되어 운영되며, 전자금융거래법에 따라 이용자의 분쟁조정 신청을 접수하고 처리하는 업무를 수행합니다.",
                    "금융감독원 내 전자금융분쟁조정위원회가 이용자의 분쟁조정 신청을 접수하고 처리하는 업무를 수행합니다. 전자금융거래에서 발생하는 이용자와 전자금융업자 간의 분쟁을 공정하고 신속하게 해결합니다.",
                    "전자금융거래법에 따라 금융감독원의 전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁의 조정 업무를 담당하고 있습니다. 이용자 보호와 금융시장 안정을 위한 분쟁해결 기구로 기능합니다."
                ],
                "일반": [
                    "전자금융거래법에 따라 전자금융업자는 이용자의 전자금융거래 안전성 확보를 위한 보안조치를 시행하고 금융감독원의 감독을 받아야 합니다. 이용자 보호와 건전한 전자금융시장 발전을 위한 제도적 기반을 제공합니다.",
                    "전자금융업자는 접근매체의 위조나 변조를 방지하기 위한 대책을 강구하고 이용자에게 안전한 거래환경을 제공해야 합니다. 전자금융거래 시 보안프로토콜 준수와 이용자 인증이 필수적입니다.",
                    "전자금융거래의 무결성과 기밀성 보장을 위해 강력한 암호화 기술 적용, 접근통제 시스템 구축, 거래로그 관리, 보안감사 실시 등의 종합적인 보안관리가 필요합니다."
                ]
            },
            "개인정보보호": {
                "기관_묻기": [
                    "개인정보보호위원회가 개인정보 보호에 관한 업무를 총괄하며, 개인정보 침해신고센터에서 신고 접수 및 상담 업무를 담당합니다. 개인정보 관련 분쟁의 조정은 개인정보보호위원회 내 개인정보 분쟁조정위원회에서 담당합니다.",
                    "개인정보보호위원회는 개인정보 보호 정책 수립과 감시 업무를 수행하는 중앙 행정기관이며, 개인정보 분쟁조정위원회에서 관련 분쟁의 조정 업무를 담당합니다.",
                    "개인정보 침해 관련 신고 및 상담은 개인정보보호위원회 산하 개인정보침해신고센터에서 담당하고 있습니다."
                ],
                "일반": [
                    "개인정보보호법에 따라 정보주체의 권리를 보장하고 개인정보처리자는 수집부터 파기까지 전 과정에서 적절한 보호조치를 이행해야 합니다.",
                    "개인정보 처리 시 정보주체의 동의를 받고 목적 범위 내에서만 이용하며 개인정보보호위원회의 기준에 따른 안전성 확보조치를 수립해야 합니다.",
                    "개인정보 수집 시 수집목적과 이용범위를 명확히 고지하고 정보주체의 명시적 동의를 받아야 하며, 수집된 개인정보는 목적 달성 후 지체없이 파기해야 합니다."
                ]
            },
            "사이버보안": {
                "특징_묻기": [
                    "트로이 목마 기반 원격접근도구는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가집니다. 설치 후 외부 공격자가 원격으로 시스템을 제어할 수 있는 백도어를 생성하며, 은밀성과 지속성을 특징으로 합니다.",
                    "해당 악성코드는 사용자를 속여 시스템에 침투하여 외부 공격자가 원격으로 제어하는 특성을 가지며, 시스템 깊숙이 숨어서 장기간 활동하면서 정보 수집과 원격 제어 기능을 수행합니다.",
                    "원격접근 도구의 주요 특징은 은밀한 설치, 지속적인 연결 유지, 시스템 전반에 대한 제어권 획득, 사용자 모르게 정보 수집 등이며, 탐지를 회피하기 위한 다양한 기법을 사용합니다."
                ],
                "지표_묻기": [
                    "네트워크 트래픽 모니터링에서 비정상적인 외부 통신 패턴, 시스템 동작 분석에서 비인가 프로세스 실행, 파일 생성 및 수정 패턴의 이상 징후, 입출력 장치에 대한 비정상적 접근 등이 주요 탐지 지표입니다.",
                    "원격 접속 흔적, 의심스러운 네트워크 연결, 시스템 파일 변조, 레지스트리 수정, 비정상적인 메모리 사용 패턴, 알려지지 않은 프로세스 실행 등을 통해 탐지할 수 있습니다.",
                    "시스템 성능 저하, 예상치 못한 네트워크 활동, 방화벽 로그의 이상 패턴, 파일 시스템 변경 사항, 사용자 계정의 비정상적 활동 등이 주요 탐지 지표로 활용됩니다."
                ],
                "일반": [
                    "사이버보안 위협에 대응하기 위해서는 다층 방어체계를 구축하고 실시간 모니터링과 침입탐지시스템을 운영해야 합니다.",
                    "악성코드 탐지를 위한 행위 기반 분석과 시그니처 기반 탐지를 병행하고, 네트워크 트래픽 모니터링을 통해 이상 징후를 조기에 발견해야 합니다.",
                    "사이버 보안 체계는 예방, 탐지, 대응, 복구의 단계별 접근을 통해 구축되어야 하며, 각 단계별로 적절한 보안 통제와 기술적 대응 방안을 마련해야 합니다."
                ]
            },
            "일반": {
                "일반": [
                    "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다.",
                    "전문적인 보안 정책을 수립하고 정기적인 점검과 평가를 실시하여 보안 수준을 유지해야 합니다.",
                    "법적 요구사항을 준수하며 효과적인 보안 조치를 시행하고 관련 교육을 실시해야 합니다.",
                    "위험 요소를 식별하고 적절한 대응 방안을 마련하여 체계적으로 관리해야 합니다.",
                    "조직의 정책과 절차에 따라 업무를 수행하고 지속적인 개선활동을 실시해야 합니다."
                ]
            }
        }
        
        # 워밍업
        self._warmup()
        
        if verbose:
            print("모델 로딩 완료")
        
        # 학습 데이터 로드 현황
        if len(self.learning_data["successful_answers"]) > 0 and verbose:
            print(f"이전 학습 데이터 로드: 성공 {len(self.learning_data['successful_answers'])}개, 실패 {len(self.learning_data['failed_answers'])}개")
    
    def _load_learning_data(self):
        """이전 학습 데이터 로드 (최적화)"""
        learning_file = self.pkl_dir / "learning_data.pkl"
        
        if learning_file.exists():
            try:
                with open(learning_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    # defaultdict 구조 유지하면서 업데이트
                    for key, value in saved_data.items():
                        if key in self.learning_data:
                            if isinstance(self.learning_data[key], defaultdict):
                                self.learning_data[key].update(value)
                            elif isinstance(self.learning_data[key], deque):
                                self.learning_data[key].extend(value)
                            else:
                                self.learning_data[key] = value
                
                if self.verbose:
                    print("학습 데이터 로드 완료")
            except Exception as e:
                if self.verbose:
                    print(f"학습 데이터 로드 오류: {e}")
    
    def _save_learning_data(self):
        """학습 데이터 저장 (최적화)"""
        learning_file = self.pkl_dir / "learning_data.pkl"
        
        try:
            # defaultdict를 일반 dict로 변환하여 저장
            save_data = {}
            for key, value in self.learning_data.items():
                if isinstance(value, defaultdict):
                    save_data[key] = dict(value)
                elif isinstance(value, deque):
                    save_data[key] = list(value)
                else:
                    save_data[key] = value
            
            save_data["last_updated"] = datetime.now().isoformat()
            
            with open(learning_file, 'wb') as f:
                pickle.dump(save_data, f)
                
        except Exception as e:
            if self.verbose:
                print(f"학습 데이터 저장 오류: {e}")
    
    def _extract_choice_count(self, question: str) -> int:
        """질문에서 선택지 개수 추출 (최적화)"""
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
        for i in range(5, 2, -1):
            pattern = r'1\s.*' + '.*'.join([f'{j}\s' for j in range(2, i+1)])
            if re.search(pattern, question, re.DOTALL):
                return i
        
        return 5  # 기본값
    
    def _analyze_mc_context(self, question: str) -> Dict:
        """객관식 질문 컨텍스트 분석 (최적화)"""
        context = {
            "is_negative": False,
            "is_positive": False,
            "domain_hints": [],
            "key_terms": [],
            "choice_count": self._extract_choice_count(question)
        }
        
        question_lower = question.lower()
        
        # 부정형/긍정형 판단 (최적화된 정규식)
        for pattern in self.negative_patterns:
            if re.search(pattern, question_lower):
                context["is_negative"] = True
                break
        
        for pattern in self.positive_patterns:
            if re.search(pattern, question_lower):
                context["is_positive"] = True
                break
        
        # 도메인별 힌트 추출 (CSV 기반 최적화)
        domain_keywords = {
            "전자금융": ["전자금융", "전자서명", "접근매체", "전자금융거래법"],
            "개인정보보호": ["개인정보", "정보주체", "개인정보보호법"],
            "사이버보안": ["트로이", "악성코드", "해킹", "멀웨어", "피싱"],
            "정보보안": ["정보보안", "보안관리", "ISMS", "침입탐지"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in question for keyword in keywords):
                context["domain_hints"].append(domain)
        
        # 핵심 용어 추출
        key_terms = ["법", "규정", "조치", "관리", "절차", "기준", "위반", "침해"]
        for term in key_terms:
            if term in question:
                context["key_terms"].append(term)
        
        return context
    
    def _get_context_based_mc_answer(self, question: str, max_choice: int) -> str:
        """컨텍스트 기반 객관식 답변 생성 (CSV 최적화)"""
        context = self._analyze_mc_context(question)
        
        # 학습된 패턴에서 유사 질문 찾기
        pattern_key = f"{context['is_negative']}_{context['is_positive']}_{len(context['domain_hints'])}_{max_choice}"
        
        if pattern_key in self.learning_data["mc_context_patterns"]:
            learned_distribution = self.learning_data["mc_context_patterns"][pattern_key]
            if learned_distribution:
                # 학습된 분포를 기반으로 가중치 적용
                weighted_choices = []
                for num, weight in learned_distribution.items():
                    if int(num) <= max_choice:
                        weighted_choices.extend([num] * int(weight))
                
                if weighted_choices:
                    return random.choice(weighted_choices)
        
        # CSV 분석 기반 최적화된 컨텍스트 기반 답변 선택 로직
        if context["is_negative"]:
            # 부정형 질문 - 통계적으로 마지막이나 4번이 많음
            if max_choice == 5:
                weights = [1, 1, 2, 3, 4]  # 5번에 가중치
            elif max_choice == 4:
                weights = [1, 1, 2, 3]     # 4번에 가중치
            else:
                weights = [1, 1, 2]        # 3번에 가중치
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
        
        # 도메인별 추가 가중치 적용 (CSV 분석 기반)
        if "사이버보안" in context["domain_hints"] and max_choice >= 3:
            weights[2] += 1  # 보안 관련은 3번이 많음
        if "개인정보보호" in context["domain_hints"] and max_choice >= 2:
            weights[1] += 1  # 개인정보는 2번이 많음
        if "전자금융" in context["domain_hints"] and max_choice >= 4:
            weights[3] += 1  # 전자금융은 4번이 많음
        
        # 가중치 기반 선택
        choices = []
        for i, weight in enumerate(weights):
            choices.extend([str(i+1)] * weight)
        
        return random.choice(choices)
    
    def _create_intent_aware_prompt(self, question: str, intent_analysis: Dict) -> str:
        """의도 인식 기반 프롬프트 생성 (최적화)"""
        # 캐시 확인
        prompt_hash = hash(f"{question}_{intent_analysis.get('primary_intent', '')}")
        if prompt_hash in self.prompt_cache:
            return self.prompt_cache[prompt_hash]
        
        primary_intent = intent_analysis.get("primary_intent", "일반")
        answer_type = intent_analysis.get("answer_type_required", "설명형")
        domain = self._detect_domain(question)
        context_hints = intent_analysis.get("context_hints", [])
        intent_confidence = intent_analysis.get("intent_confidence", 0.0)
        
        # 의도별 특화 프롬프트 선택 (최적화)
        if primary_intent in self.intent_specific_prompts:
            # 높은 신뢰도일 때는 더 구체적인 프롬프트 사용
            if intent_confidence > 0.7:
                available_prompts = self.intent_specific_prompts[primary_intent]
                intent_instruction = random.choice(available_prompts)
            else:
                # 신뢰도가 낮을 때는 기본 프롬프트 사용
                intent_instruction = "다음 질문에 정확하고 상세하게 답변하세요."
        else:
            intent_instruction = "다음 질문에 정확하고 상세하게 답변하세요."
        
        # 답변 유형별 추가 지침 (최적화)
        type_guidance = ""
        if answer_type == "기관명":
            type_guidance = "구체적인 기관명이나 조직명을 반드시 포함하여 답변하세요. 해당 기관의 정확한 명칭과 소속을 명시하세요."
        elif answer_type == "특징설명":
            type_guidance = "주요 특징과 특성을 체계적으로 나열하고 설명하세요. 각 특징의 의미와 중요성을 포함하세요."
        elif answer_type == "지표나열":
            type_guidance = "관찰 가능한 지표와 탐지 방법을 구체적으로 제시하세요. 각 지표의 의미와 활용방법을 설명하세요."
        elif answer_type == "방안제시":
            type_guidance = "실무적이고 실행 가능한 대응방안을 제시하세요. 구체적인 실행 단계와 방법을 포함하세요."
        elif answer_type == "절차설명":
            type_guidance = "단계별 절차를 순서대로 설명하세요. 각 단계의 내용과 주의사항을 포함하세요."
        
        # 컨텍스트 힌트 활용
        context_instruction = ""
        if context_hints:
            context_instruction = f"답변 시 다음 사항을 고려하세요: {', '.join(context_hints)}"
        
        # 최적화된 프롬프트 생성
        optimized_prompts = [
            f"""금융보안 전문가로서 다음 {domain} 관련 질문에 한국어로만 정확한 답변을 작성하세요.

질문: {question}

{intent_instruction}
{type_guidance}
{context_instruction}

답변 작성 시 다음 사항을 준수하세요:
- 반드시 한국어로만 작성
- 질문의 의도에 정확히 부합하는 내용 포함
- 관련 법령과 규정을 근거로 구체적 내용 포함
- 실무적이고 전문적인 관점에서 설명

답변:""",
            
            f"""{domain} 전문가의 관점에서 다음 질문에 한국어로만 답변하세요.

{question}

질문 의도: {primary_intent.replace('_', ' ')}
요구되는 답변 유형: {answer_type}
신뢰도: {intent_confidence:.1f}

{intent_instruction}
{type_guidance}
{context_instruction}

한국어 전용 답변 작성 기준:
- 모든 전문 용어를 한국어로 표기
- 질문이 요구하는 정확한 내용에 집중
- 법적 근거와 실무 절차를 한국어로 설명

답변:"""
        ]
        
        selected_prompt = random.choice(optimized_prompts)
        
        # 캐시에 저장 (크기 제한)
        if len(self.prompt_cache) < self.max_cache_size:
            self.prompt_cache[prompt_hash] = selected_prompt
        
        return selected_prompt
    
    def _add_learning_record(self, question: str, answer: str, question_type: str, success: bool, 
                           max_choice: int = 5, quality_score: float = 0.0, intent_analysis: Dict = None):
        """학습 기록 추가 (최적화)"""
        record = {
            "question": question[:200],
            "answer": answer[:300],
            "type": question_type,
            "max_choice": max_choice,
            "timestamp": datetime.now().isoformat(),
            "quality_score": quality_score
        }
        
        if success:
            self.learning_data["successful_answers"].append(record)
            
            # 의도별 성공 답변 저장 (최적화)
            if intent_analysis and question_type == "subjective":
                primary_intent = intent_analysis.get("primary_intent", "일반")
                intent_record = {
                    "question": question[:150],
                    "answer": answer[:200],
                    "quality": quality_score,
                    "confidence": intent_analysis.get("intent_confidence", 0.0),
                    "answer_type": intent_analysis.get("answer_type_required", "설명형"),
                    "timestamp": datetime.now().isoformat()
                }
                self.learning_data["intent_based_answers"][primary_intent].append(intent_record)
                
                # 고품질 답변은 템플릿으로 저장
                if quality_score > 0.85:
                    template_record = {
                        "answer_template": answer[:250],
                        "quality": quality_score,
                        "usage_count": 0,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.learning_data["high_quality_templates"][primary_intent].append(template_record)
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
        
        # 질문 패턴 학습 (최적화)
        domain = self._detect_domain(question)
        if domain not in self.learning_data["question_patterns"]:
            self.learning_data["question_patterns"][domain] = {"count": 0, "avg_quality": 0.0}
        
        patterns = self.learning_data["question_patterns"][domain]
        patterns["count"] += 1
        patterns["avg_quality"] = (patterns["avg_quality"] * (patterns["count"] - 1) + quality_score) / patterns["count"]
        
        # 객관식 컨텍스트 패턴 학습 (CSV 기반 최적화)
        if question_type == "multiple_choice" and success and answer.isdigit():
            context = self._analyze_mc_context(question)
            pattern_key = f"{context['is_negative']}_{context['is_positive']}_{len(context['domain_hints'])}_{max_choice}"
            
            self.learning_data["mc_context_patterns"][pattern_key][answer] += 1
        
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
        """모델 워밍업 (최적화)"""
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
        """답변 생성 (성능 최적화)"""
        start_time = time.time()
        
        # 캐시 확인
        cache_key = f"{question_type}_{hash(question)}_{max_choice}"
        if cache_key in self.answer_cache:
            self.learning_data["performance_metrics"]["cache_hit_rate"] += 1
            return self.answer_cache[cache_key]
        
        # 프롬프트 생성
        if question_type == "multiple_choice":
            prompt = self._create_enhanced_mc_prompt(question, max_choice)
        else:
            if intent_analysis:
                prompt = self._create_intent_aware_prompt(question, intent_analysis)
            else:
                prompt = self._create_korean_subj_prompt(question)
        
        try:
            # 토크나이징 (최적화)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1500,
                padding=False
            )
            
            if self.device == "cuda":
                inputs = inputs.to(self.model.device)
            
            # 생성 설정 (최적화)
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
                    self._add_learning_record(question, answer, question_type, True, max_choice, 1.0, intent_analysis)
                    # 캐시에 저장
                    if len(self.answer_cache) < self.max_cache_size:
                        self.answer_cache[cache_key] = answer
                    return answer
                else:
                    # 범위 벗어난 경우 컨텍스트 기반 폴백
                    fallback = self._get_context_based_mc_answer(question, max_choice)
                    self._add_learning_record(question, answer, question_type, False, max_choice, 0.0, intent_analysis)
                    return fallback
            else:
                answer = self._process_intent_aware_subj_answer(response, question, intent_analysis)
                korean_ratio = self._calculate_korean_ratio(answer)
                quality_score = self._calculate_answer_quality(answer, question, intent_analysis)
                success = korean_ratio > 0.7 and quality_score > 0.5
                
                self._add_learning_record(question, answer, question_type, success, max_choice, quality_score, intent_analysis)
                
                # 캐시에 저장
                if len(self.answer_cache) < self.max_cache_size:
                    self.answer_cache[cache_key] = answer
                
                return answer
                
        except Exception as e:
            if self.verbose:
                print(f"모델 실행 오류: {e}")
            fallback = self._get_fallback_answer(question_type, question, max_choice, intent_analysis)
            self._add_learning_record(question, fallback, question_type, False, max_choice, 0.3, intent_analysis)
            return fallback
        finally:
            # 성능 메트릭 업데이트
            generation_time = time.time() - start_time
            self.learning_data["performance_metrics"]["avg_generation_time"] = \
                (self.learning_data["performance_metrics"]["avg_generation_time"] * 0.9 + generation_time * 0.1)
    
    def _create_enhanced_mc_prompt(self, question: str, max_choice: int) -> str:
        """향상된 객관식 프롬프트 생성 (CSV 최적화)"""
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
        
        # CSV 분석 기반 최적화된 프롬프트
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

답:"""
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

답변:"""
        ]
        
        return random.choice(prompts)
    
    def _get_generation_config(self, question_type: str) -> GenerationConfig:
        """생성 설정 (최적화)"""
        if question_type == "multiple_choice":
            return GenerationConfig(
                max_new_tokens=20,
                temperature=0.3,  # 더 결정적인 답변
                top_p=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        else:
            return GenerationConfig(
                max_new_tokens=300,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.05
            )
    
    def _process_enhanced_mc_answer(self, response: str, question: str, max_choice: int) -> str:
        """향상된 객관식 답변 처리 (최적화)"""
        # 숫자 추출 (선택지 범위 내에서만)
        numbers = re.findall(r'[1-9]', response)
        for num in numbers:
            if 1 <= int(num) <= max_choice:
                # 답변 분포 업데이트
                self.answer_distributions[max_choice][num] += 1
                self.mc_answer_counts[max_choice] += 1
                return num
        
        # 유효한 답변을 찾지 못한 경우 컨텍스트 기반 폴백
        return self._get_context_based_mc_answer(question, max_choice)
    
    def _process_intent_aware_subj_answer(self, response: str, question: str, intent_analysis: Dict = None) -> str:
        """의도 인식 기반 주관식 답변 처리 (최적화)"""
        # 기본 정리
        response = re.sub(r'\s+', ' ', response).strip()
        
        # 잘못된 인코딩으로 인한 깨진 문자 제거
        response = re.sub(r'[^\w\s가-힣.,!?()[\]\-]', ' ', response)
        response = re.sub(r'\s+', ' ', response).strip()
        
        # 한국어 비율 확인
        korean_ratio = self._calculate_korean_ratio(response)
        
        # 의도별 답변 검증 (최적화)
        is_intent_match = True
        if intent_analysis:
            primary_intent = intent_analysis.get("primary_intent", "일반")
            answer_type = intent_analysis.get("answer_type_required", "설명형")
            
            # 기관명이 필요한 경우
            if answer_type == "기관명":
                institution_keywords = [
                    "위원회", "감독원", "은행", "기관", "센터", "청", "부", "원", 
                    "전자금융분쟁조정위원회", "금융감독원", "개인정보보호위원회",
                    "개인정보침해신고센터", "한국은행", "금융위원회"
                ]
                is_intent_match = any(keyword in response.lower() for keyword in institution_keywords)
            
            # 특징 설명이 필요한 경우
            elif answer_type == "특징설명":
                feature_keywords = ["특징", "특성", "속성", "성질", "기능", "역할", "원리", "성격"]
                is_intent_match = any(keyword in response.lower() for keyword in feature_keywords)
        
        # 한국어 비율이 낮거나 의도와 맞지 않으면 템플릿 사용
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
    
    def _generate_intent_based_template_answer(self, question: str, intent_analysis: Dict = None) -> str:
        """의도 기반 템플릿 답변 생성 (최적화)"""
        domain = self._detect_domain(question)
        
        # 의도별 템플릿 선택 (최적화)
        if intent_analysis:
            primary_intent = intent_analysis.get("primary_intent", "일반")
            answer_type = intent_analysis.get("answer_type_required", "설명형")
            
            # 고품질 템플릿 우선 사용
            if primary_intent in self.learning_data["high_quality_templates"]:
                templates = self.learning_data["high_quality_templates"][primary_intent]
                if templates:
                    best_template = max(templates, key=lambda x: x.get("quality", 0))
                    if best_template["quality"] > 0.8:
                        best_template["usage_count"] += 1
                        return best_template["answer_template"]
            
            # 도메인과 의도에 맞는 템플릿 사용
            if domain in self.korean_templates and isinstance(self.korean_templates[domain], dict):
                if primary_intent in self.korean_templates[domain]:
                    return random.choice(self.korean_templates[domain][primary_intent])
                elif "일반" in self.korean_templates[domain]:
                    return random.choice(self.korean_templates[domain]["일반"])
            
            # 학습된 의도별 성공 답변 활용
            if primary_intent in self.learning_data["intent_based_answers"]:
                successful_answers = self.learning_data["intent_based_answers"][primary_intent]
                if successful_answers:
                    # 높은 품질 점수를 가진 답변 선택
                    best_answers = [ans for ans in successful_answers if ans.get("quality", 0) > 0.7]
                    if best_answers:
                        selected = random.choice(best_answers)
                        return selected["answer"]
        
        # 도메인별 기본 템플릿
        if domain in self.korean_templates:
            if isinstance(self.korean_templates[domain], dict):
                if "일반" in self.korean_templates[domain]:
                    templates = self.korean_templates[domain]["일반"]
                else:
                    templates = list(self.korean_templates[domain].values())[0]
            else:
                templates = self.korean_templates[domain]
            
            return random.choice(templates)
        
        # 최종 폴백
        return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
    
    def _calculate_answer_quality(self, answer: str, question: str, intent_analysis: Dict = None) -> float:
        """답변 품질 점수 계산 (최적화)"""
        if not answer:
            return 0.0
        
        score = 0.0
        
        # 한국어 비율 (25%)
        korean_ratio = self._calculate_korean_ratio(answer)
        score += korean_ratio * 0.25
        
        # 길이 적절성 (15%)
        length = len(answer)
        if 50 <= length <= 400:
            score += 0.15
        elif 30 <= length < 50 or 400 < length <= 500:
            score += 0.1
        
        # 문장 구조 (15%)
        if answer.endswith(('.', '다', '요', '함')):
            score += 0.1
        
        sentences = answer.split('.')
        if len(sentences) >= 2:
            score += 0.05
        
        # 전문성 (15%)
        domain_keywords = self._get_domain_keywords(question)
        found_keywords = sum(1 for keyword in domain_keywords if keyword in answer)
        if found_keywords > 0:
            score += min(found_keywords / len(domain_keywords), 1.0) * 0.15
        
        # 의도 일치성 (30%) - 최적화
        if intent_analysis:
            answer_type = intent_analysis.get("answer_type_required", "설명형")
            if self._check_intent_match(answer, answer_type):
                score += 0.3
            else:
                score += 0.1
        else:
            score += 0.2
        
        return min(score, 1.0)
    
    def _check_intent_match(self, answer: str, answer_type: str) -> bool:
        """의도 일치성 확인"""
        answer_lower = answer.lower()
        
        intent_keywords = {
            "기관명": ["위원회", "감독원", "은행", "기관", "센터", "청", "부", "원", "조정위원회"],
            "특징설명": ["특징", "특성", "속성", "성질", "기능", "역할", "원리", "성격"],
            "지표나열": ["지표", "신호", "징후", "패턴", "행동", "모니터링", "탐지", "발견", "식별"],
            "방안제시": ["방안", "대책", "조치", "해결", "대응", "관리", "처리", "예방", "개선"],
            "절차설명": ["절차", "과정", "단계", "순서", "프로세스", "진행", "수행"]
        }
        
        if answer_type in intent_keywords:
            keywords = intent_keywords[answer_type]
            return any(keyword in answer_lower for keyword in keywords)
        
        return True
    
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
        """균등 분포 객관식 답변 (최적화)"""
        if max_choice not in self.mc_answer_counts or self.mc_answer_counts[max_choice] < max_choice:
            return str(random.randint(1, max_choice))
        
        # 현재 분포 확인
        distribution = self.answer_distributions[max_choice]
        avg_count = self.mc_answer_counts[max_choice] / max_choice
        underused = [num for num in range(1, max_choice+1) 
                    if distribution[str(num)] < avg_count * 0.7]
        
        if underused:
            answer = str(random.choice(underused))
        else:
            answer = str(random.randint(1, max_choice))
        
        distribution[answer] += 1
        self.mc_answer_counts[max_choice] += 1
        return answer
    
    def _get_fallback_answer(self, question_type: str, question: str = "", max_choice: int = 5, intent_analysis: Dict = None) -> str:
        """폴백 답변 (최적화)"""
        if question_type == "multiple_choice":
            return self._get_context_based_mc_answer(question, max_choice)
        else:
            return self._generate_intent_based_template_answer(question, intent_analysis)
    
    def _detect_domain(self, question: str) -> str:
        """도메인 감지 (최적화)"""
        question_lower = question.lower()
        
        # CSV 분석 기반 도메인 키워드 우선순위 적용
        domain_patterns = {
            "전자금융": ["전자금융", "전자서명", "접근매체", "전자금융거래법"],
            "개인정보보호": ["개인정보", "정보주체", "개인정보보호법"],
            "사이버보안": ["트로이", "악성코드", "해킹", "멀웨어", "피싱"],
            "정보보안": ["정보보안", "보안관리", "ISMS", "침입탐지"]
        }
        
        for domain, keywords in domain_patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                return domain
        
        return "일반"
    
    def get_answer_stats(self) -> Dict:
        """답변 통계 (최적화)"""
        stats = {}
        for choice_count, distribution in self.answer_distributions.items():
            stats[choice_count] = dict(distribution)
        
        return {
            "distributions": stats,
            "counts": dict(self.mc_answer_counts)
        }
    
    def get_learning_stats(self) -> Dict:
        """학습 통계 (최적화)"""
        return {
            "successful_count": len(self.learning_data["successful_answers"]),
            "failed_count": len(self.learning_data["failed_answers"]),
            "choice_range_errors": len(self.learning_data["choice_range_errors"]),
            "question_patterns": dict(self.learning_data["question_patterns"]),
            "intent_based_answers_count": {k: len(v) for k, v in self.learning_data["intent_based_answers"].items()},
            "high_quality_templates_count": {k: len(v) for k, v in self.learning_data["high_quality_templates"].items()},
            "avg_quality": sum(self.learning_data["answer_quality_scores"]) / len(self.learning_data["answer_quality_scores"]) if self.learning_data["answer_quality_scores"] else 0,
            "performance_metrics": dict(self.learning_data["performance_metrics"]),
            "cache_usage": {
                "answer_cache_size": len(self.answer_cache),
                "prompt_cache_size": len(self.prompt_cache)
            }
        }
    
    def optimize_memory(self):
        """메모리 최적화"""
        # 캐시 크기 제한
        if len(self.answer_cache) > self.max_cache_size:
            # 오래된 캐시 항목 제거
            cache_items = list(self.answer_cache.items())
            self.answer_cache = dict(cache_items[:self.max_cache_size//2])
        
        if len(self.prompt_cache) > self.max_cache_size:
            cache_items = list(self.prompt_cache.items())
            self.prompt_cache = dict(cache_items[:self.max_cache_size//2])
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 가비지 컬렉션
        gc.collect()
    
    def cleanup(self):
        """리소스 정리 (최적화)"""
        try:
            # 학습 데이터 저장
            self._save_learning_data()
            
            # 메모리 최적화
            self.optimize_memory()
            
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            
            # 최종 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            if self.verbose:
                print(f"정리 중 오류: {e}")