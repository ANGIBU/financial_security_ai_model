# prompt_enhancer.py

import re
import random
import hashlib
from typing import Dict, List


class PromptEnhancer:
    """프롬프트 구성 및 Few-shot 예시 관리"""
    
    def __init__(self):
        self._initialize_few_shot_examples()
        self._initialize_prompt_templates()
        self.used_examples_cache = {}
        self.prompt_history = []
        
    def _initialize_few_shot_examples(self):
        """Few-shot 예시 초기화"""
        
        self.few_shot_examples = {
            "사이버보안": {
                "multiple_choice": [
                    {
                        "question": "다음 중 트로이 목마의 주요 특징으로 가장 적절한 것은?\n1 자가 복제 기능\n2 정상 프로그램 위장\n3 네트워크 속도 저하\n4 파일 암호화\n5 화면 잠금",
                        "answer": "2",
                        "reasoning": "트로이 목마는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 것이 주요 특징입니다.",
                        "domain_focus": ["트로이", "특징", "위장"],
                        "difficulty": "중급"
                    },
                    {
                        "question": "SBOM을 금융권에서 활용하는 가장 적절한 이유는?\n1 데이터 백업\n2 네트워크 모니터링\n3 접근 권한 관리\n4 암호화 강화\n5 소프트웨어 공급망 보안",
                        "answer": "5",
                        "reasoning": "SBOM은 소프트웨어 구성 요소의 투명성을 제공하여 공급망 보안을 강화하는 목적으로 활용됩니다.",
                        "domain_focus": ["SBOM", "활용", "공급망"],
                        "difficulty": "고급"
                    }
                ],
                "subjective": [
                    {
                        "question": "트로이 목마(Trojan) 기반 원격제어 악성코드(RAT)의 특징과 주요 탐지 지표를 설명하세요.",
                        "answer": "트로이 목마 기반 원격제어 악성코드는 정상 프로그램으로 위장하여 시스템에 침투하고 외부 공격자가 원격으로 시스템을 제어할 수 있도록 하는 특성을 가집니다. 주요 탐지 지표로는 비정상적인 네트워크 통신 패턴, 비인가 프로세스 실행, 파일 시스템 변경, 레지스트리 수정 등이 있으며, 실시간 모니터링과 행동 분석을 통한 종합적 탐지가 필요합니다.",
                        "domain_focus": ["트로이", "특징", "탐지지표"],
                        "answer_type": "특징설명",
                        "quality_score": 0.9
                    },
                    {
                        "question": "딥페이크 기술 악용에 대비한 금융권의 대응 방안을 기술하세요.",
                        "answer": "딥페이크 기술 악용에 대비하여 금융권에서는 다층 방어체계 구축, 딥보이스 탐지 기술 개발 및 도입, 생체인증과 다중 인증 체계를 통한 신원 검증 강화, 직원 교육 및 고객 인식 제고를 통한 선제적 보안 대응 방안을 수립해야 합니다.",
                        "domain_focus": ["딥페이크", "대응", "방안"],
                        "answer_type": "방안제시",
                        "quality_score": 0.85
                    }
                ]
            },
            "전자금융": {
                "multiple_choice": [
                    {
                        "question": "한국은행이 금융통화위원회의 요청에 따라 전자금융업자에게 자료제출을 요구할 수 있는 경우로 가장 적절한 것은?\n1 보안 시스템 점검\n2 고객 정보 확인\n3 경영 실적 조사\n4 통화신용정책 수행\n5 시장 동향 파악",
                        "answer": "4",
                        "reasoning": "한국은행법에 따라 통화신용정책의 수행 및 지급결제제도의 원활한 운영을 위해 자료제출을 요구할 수 있습니다.",
                        "domain_focus": ["한국은행", "자료제출", "통화신용정책"],
                        "difficulty": "고급"
                    }
                ],
                "subjective": [
                    {
                        "question": "전자금융거래법에 따라 이용자가 금융 분쟁조정을 신청할 수 있는 기관을 기술하세요.",
                        "answer": "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당하며, 금융감독원 내에 설치되어 전자금융거래법 제28조에 근거하여 이용자와 전자금융업자 간의 분쟁을 공정하고 신속하게 해결하는 역할을 수행합니다.",
                        "domain_focus": ["분쟁조정", "기관", "전자금융거래법"],
                        "answer_type": "기관명",
                        "quality_score": 0.9
                    },
                    {
                        "question": "금융회사가 정보보호 예산을 관리할 때, 전자금융감독규정상 정보기술부문 인력 및 예산의 기준 비율은 얼마인가요?",
                        "answer": "전자금융감독규정 제16조에 따라 금융회사는 정보기술부문 인력을 총 인력의 5% 이상, 정보기술부문 예산을 총 예산의 7% 이상 정보보호 업무에 배정해야 합니다. 다만 회사 규모, 업무 특성, 정보기술 위험수준 등을 고려하여 금융감독원장이 별도로 정할 수 있습니다.",
                        "domain_focus": ["정보기술부문", "비율", "예산"],
                        "answer_type": "수치설명",
                        "quality_score": 0.95
                    }
                ]
            },
            "개인정보보호": {
                "multiple_choice": [
                    {
                        "question": "만 14세 미만 아동의 개인정보 처리를 위해 필요한 절차로 가장 적절한 것은?\n1 본인의 직접 동의\n2 법정대리인의 동의\n3 학교의 승인\n4 관할 기관 허가\n5 보호자 확인서",
                        "answer": "2",
                        "reasoning": "개인정보보호법 제22조에 따라 만 14세 미만 아동의 개인정보 처리에는 법정대리인의 동의가 필요합니다.",
                        "domain_focus": ["만14세", "아동", "법정대리인"],
                        "difficulty": "중급"
                    }
                ],
                "subjective": [
                    {
                        "question": "개인정보 접근 권한 검토는 어떻게 수행해야 하며, 그 목적은 무엇인가요?",
                        "answer": "개인정보 접근 권한 검토는 업무상 필요한 최소한의 권한만을 부여하는 최소권한 원칙에 따라 정기적으로 수행해야 하며, 불필요한 권한은 즉시 회수하여 개인정보 오남용을 방지하고 정보보안을 강화하는 것이 목적입니다.",
                        "domain_focus": ["접근권한", "검토", "최소권한"],
                        "answer_type": "절차설명",
                        "quality_score": 0.85
                    }
                ]
            },
            "정보보안": {
                "multiple_choice": [
                    {
                        "question": "재해 복구 계획 수립 시 고려 요소 중 옳지 않은 것은?\n1 복구 절차 수립\n2 비상연락체계 구축\n3 개인정보 파기 절차\n4 복구 목표시간 설정\n5 백업 시스템 구축",
                        "answer": "3",
                        "reasoning": "개인정보 파기 절차는 재해 복구 계획과 직접적인 관련이 없으며, 복구 관련 요소가 아닙니다.",
                        "domain_focus": ["재해복구", "계획수립", "파기절차"],
                        "difficulty": "중급"
                    }
                ],
                "subjective": [
                    {
                        "question": "정보보호의 3대 요소에 해당하는 보안 목표를 3가지 기술하세요.",
                        "answer": "정보보호의 3대 요소는 기밀성(Confidentiality), 무결성(Integrity), 가용성(Availability)으로 구성됩니다. 기밀성은 인가된 사용자만이 정보에 접근할 수 있도록 하는 것이며, 무결성은 정보의 정확성과 완전성을 보장하는 것입니다. 가용성은 인가된 사용자가 필요할 때 언제든지 정보와 자원에 접근할 수 있도록 보장하는 것입니다.",
                        "domain_focus": ["3대요소", "보안목표"],
                        "answer_type": "정의설명",
                        "quality_score": 0.9
                    }
                ]
            },
            "정보통신": {
                "multiple_choice": [
                    {
                        "question": "집적된 정보통신시설의 보호와 관련하여 정보통신서비스 제공의 중단이 발생했을 때, 과학기술정보통신부장관에게 보고해야 하는 사항으로 옳지 않은 것은?\n1 정보통신서비스 제공의 중단이 발생한 일시 및 장소\n2 정보통신서비스 제공의 중단이 발생한 원인에 대한 법적 책임\n3 정보통신서비스 제공의 중단이 발생한 원인 및 피해내용\n4 응급조치 사항",
                        "answer": "2",
                        "reasoning": "정보통신서비스 제공 중단 발생 시 과학기술정보통신부장관에게 보고해야 하는 사항에는 법적 책임이 포함되지 않습니다.",
                        "domain_focus": ["정보통신서비스", "중단", "보고사항"],
                        "difficulty": "고급"
                    }
                ],
                "subjective": [
                    {
                        "question": "정보통신시설의 중단 발생 시 과학기술정보통신부장관에게 보고해야 하는 사항을 설명하세요.",
                        "answer": "집적된 정보통신시설의 보호와 관련하여 정보통신서비스 제공의 중단이 발생했을 때, 정보통신기반 보호법에 따라 과학기술정보통신부장관에게 보고해야 하는 사항은 중단이 발생한 일시 및 장소, 중단이 발생한 원인 및 피해내용, 응급조치 사항입니다. 다만 법적 책임에 관한 사항은 보고 대상에 해당하지 않습니다.",
                        "domain_focus": ["정보통신시설", "중단보고", "보고사항"],
                        "answer_type": "절차설명",
                        "quality_score": 0.9
                    }
                ]
            },
            "금융투자": {
                "multiple_choice": [
                    {
                        "question": "금융산업의 이해와 관련하여 금융투자업의 구분에 해당하지 않는 것은?\n1 소비자금융업\n2 투자자문업\n3 투자매매업\n4 투자중개업\n5 보험중개업",
                        "answer": "1",
                        "reasoning": "소비자금융업은 금융투자업에 해당하지 않으며, 별도의 금융업 분류에 속합니다.",
                        "domain_focus": ["금융투자업", "구분", "소비자금융업"],
                        "difficulty": "중급"
                    }
                ]
            },
            "위험관리": {
                "multiple_choice": [
                    {
                        "question": "위험 관리 계획 수립 시 고려해야 할 요소로 적절하지 않은 것은?\n1 수행인력\n2 위험 수용\n3 위험 대응 전략 선정\n4 대상\n5 기간",
                        "answer": "2",
                        "reasoning": "위험 관리 계획에서 위험 수용은 적절한 관리 요소가 아니며, 위험을 식별하고 대응하는 것이 중요합니다.",
                        "domain_focus": ["위험관리", "계획수립", "위험수용"],
                        "difficulty": "중급"
                    }
                ]
            }
        }
    
    def _initialize_prompt_templates(self):
        """프롬프트 템플릿 초기화"""
        
        self.prompt_templates = {
            "multiple_choice_basic": """다음은 금융보안 관련 객관식 문제입니다. 체계적 분석을 통해 가장 적절한 답을 선택하세요.

{few_shot_examples}

문제 분석 단계:
1. 질문의 핵심 키워드와 요구사항 파악
2. 각 선택지를 해당 법령과 규정에 따라 검토
3. 문제 유형(부정/긍정)에 따른 논리적 추론 적용
4. 도메인 전문 지식을 바탕으로 최적 답안 선택

문제: {question}

위 단계를 따라 체계적으로 분석하여 정답 번호만 제시하세요.

정답 번호: """,

            "subjective_basic": """다음은 금융보안 관련 주관식 문제입니다. 반드시 한국어로만 전문적이고 정확한 답변을 작성하세요.

답변 작성 지침:
- 모든 답변은 한국어로만 작성 (영어 사용 절대 금지)
- 관련 법령과 규정에 근거한 전문적 답변 작성
- 구체적이고 실무적인 내용 포함
- 자연스러운 한국어 문장으로 구성
- 도메인별 전문용어 적절히 활용

{few_shot_examples}

참고 정보:
{context_info}

문제: {question}

위 문제에 대해 관련 법령과 규정을 근거로 구체적이고 전문적인 한국어 답변을 작성하세요.

한국어 답변: """,

            "institution_specialized": """다음은 금융보안 관련 기관에 대한 질문입니다. 정확한 기관 정보를 바탕으로 한국어로만 답변하세요.

답변 작성 지침:
- 모든 답변은 한국어로만 작성
- 기관의 정확한 명칭과 역할 기술
- 법적 근거와 설립 배경 포함
- 구체적인 업무 범위와 절차 설명

{few_shot_examples}

기관 정보:
{institution_info}

문제: {question}

위 질문에 대해 다음 요소를 포함하여 한국어로 답변하세요:
1. 정확한 기관명과 소속 조직
2. 법적 근거와 설립 배경  
3. 주요 업무와 권한 범위
4. 관련 절차와 연락 방법

한국어 답변: """,

            "ratio_specialized": """다음은 금융보안 관련 비율에 대한 질문입니다. 구체적인 수치와 법적 근거를 포함하여 한국어로만 답변하세요.

답변 작성 지침:
- 정확한 수치와 퍼센트 명시
- 해당 법령과 조항 인용
- 예외 조건이나 특별 규정 포함
- 적용 범위와 기준 명확히 설명

{few_shot_examples}

문제: {question}

위 질문에 대해 다음 사항을 포함하여 한국어로 답변하세요:
1. 정확한 비율과 수치
2. 관련 법령과 조항 번호
3. 적용 조건 및 예외사항
4. 감독기관의 재량권과 기준

한국어 답변: """,

            "domain_specialized": {
                "사이버보안": """다음은 사이버보안 관련 문제입니다. 기술적 특성과 보안 대응 방안에 중점을 두어 답변하세요.

답변 작성 지침:
- 기술적 메커니즘과 동작 원리 상세 설명
- 실제 위협 사례와 탐지 방법 포함
- 다층 방어체계와 종합적 대응방안 제시

{few_shot_examples}

문제: {question}

사이버보안 전문가 관점에서 기술적 특징, 탐지 방법, 대응 방안을 구체적으로 설명하세요.

한국어 답변: """,

                "전자금융": """다음은 전자금융 관련 문제입니다. 전자금융거래법과 관련 규정을 근거로 답변하세요.

답변 작성 지침:
- 전자금융거래법의 구체적 조항 인용
- 이용자 보호와 업무 절차 중심 설명
- 법적 요구사항과 실무 적용 방법 제시

{few_shot_examples}

문제: {question}

전자금융거래법과 관련 규정에 근거하여 법적 요구사항과 절차를 명확히 설명하세요.

한국어 답변: """,

                "개인정보보호": """다음은 개인정보보호 관련 문제입니다. 개인정보보호법을 근거로 답변하세요.

답변 작성 지침:
- 개인정보보호법의 원칙과 절차 중심 설명
- 정보주체의 권리와 처리자의 의무 명시
- 실무 적용 사례와 주의사항 포함

{few_shot_examples}

문제: {question}

개인정보보호법에 따른 처리 원칙과 절차를 구체적으로 설명하세요.

한국어 답변: """,

                "정보보안": """다음은 정보보안 관련 문제입니다. 정보보안관리체계 관점에서 답변하세요.

답변 작성 지침:
- 정보보안관리체계의 체계적 접근 중심 설명
- 기술적·관리적·물리적 보안대책 구분 제시
- 위험 분석과 단계별 대응 방안 포함

{few_shot_examples}

문제: {question}

정보보안관리체계 구축과 운영 관점에서 체계적으로 설명하세요.

한국어 답변: """,

                "정보통신": """다음은 정보통신 관련 문제입니다. 정보통신기반 보호법을 근거로 답변하세요.

답변 작성 지침:
- 정보통신기반 보호법의 요구사항 중심 설명
- 보고 의무와 대응 절차 명확히 제시
- 관련 기관과 업무 분장 포함

{few_shot_examples}

문제: {question}

정보통신기반 보호법에 따른 요구사항과 절차를 명확히 설명하세요.

한국어 답변: """
            }
        }

    def _generate_prompt_hash(self, question: str, domain: str, question_type: str) -> str:
        """프롬프트 해시 생성"""
        try:
            combined_text = f"{question[:100]}-{domain}-{question_type}-{len(question)}"
            return hashlib.md5(combined_text.encode()).hexdigest()[:10]
        except Exception:
            return ""

    def _select_quality_examples(self, domain: str, question_type: str, question: str, count: int = 2) -> List[Dict]:
        """품질 예시 선택"""
        try:
            if domain not in self.few_shot_examples:
                return []
            
            domain_examples = self.few_shot_examples[domain]
            if question_type not in domain_examples:
                return []
            
            examples = domain_examples[question_type]
            if not examples:
                return []
            
            question_keywords = set(question.lower().split())
            
            # 예시별 점수 계산
            scored_examples = []
            for example in examples:
                score = 0
                
                # 도메인 포커스 키워드 매칭
                if "domain_focus" in example:
                    focus_matches = sum(1 for focus in example["domain_focus"] 
                                      if focus.lower() in question.lower())
                    score += focus_matches * 2.0
                
                # 답변 유형 매칭 (주관식)
                if question_type == "subjective":
                    if "answer_type" in example:
                        if self._match_answer_type(question, example["answer_type"]):
                            score += 3.0
                    
                    # 품질 점수
                    if "quality_score" in example:
                        score += example["quality_score"] * 2.0
                
                scored_examples.append((example, score))
            
            # 점수별 정렬
            scored_examples.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 선택
            selected_examples = []
            for example, score in scored_examples:
                if len(selected_examples) >= count:
                    break
                selected_examples.append(example)
            
            # 부족한 경우 추가 선택
            if len(selected_examples) < count:
                remaining = [ex for ex, _ in scored_examples if ex not in selected_examples]
                selected_examples.extend(remaining[:count - len(selected_examples)])
            
            return selected_examples[:count]
            
        except Exception as e:
            print(f"품질 예시 선택 오류: {e}")
            return examples[:count] if examples else []

    def _match_answer_type(self, question: str, answer_type: str) -> bool:
        """답변 유형 매칭"""
        question_lower = question.lower()
        
        type_patterns = {
            "기관명": ["기관", "위원회", "담당", "어디", "누구", "신청"],
            "특징설명": ["특징", "특성", "성질", "어떤", "주요"],
            "지표나열": ["지표", "징후", "탐지", "모니터링", "패턴"],
            "방안제시": ["방안", "대책", "대응", "해결", "어떻게"],
            "절차설명": ["절차", "과정", "단계", "순서", "프로세스"],
            "수치설명": ["비율", "얼마", "기준", "퍼센트", "%"],
            "정의설명": ["정의", "무엇", "개념", "의미", "뜻"]
        }
        
        if answer_type in type_patterns:
            patterns = type_patterns[answer_type]
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return True
        
        return False

    def build_few_shot_context(self, domain: str, question_type: str, question: str, count: int = 2) -> str:
        """Few-shot 예시 구성"""
        try:
            # 프롬프트 해시 생성
            prompt_hash = self._generate_prompt_hash(question, domain, question_type)
            
            # 캐시된 예시 확인 및 제외
            excluded_examples = self.used_examples_cache.get(prompt_hash, [])
            
            selected_examples = self._select_quality_examples(domain, question_type, question, count)
            
            # 제외 목록 확인
            if excluded_examples:
                available_examples = []
                for ex in selected_examples:
                    answer_signature = self._generate_example_signature(ex)
                    if answer_signature not in excluded_examples:
                        available_examples.append(ex)
                
                if available_examples:
                    selected_examples = available_examples
            
            # 사용된 예시 캐시 업데이트
            if selected_examples:
                used_signatures = [self._generate_example_signature(ex) for ex in selected_examples]
                self.used_examples_cache[prompt_hash] = used_signatures
                
                # 캐시 크기 제한
                if len(self.used_examples_cache) > 50:
                    oldest_key = list(self.used_examples_cache.keys())[0]
                    del self.used_examples_cache[oldest_key]
            
            # Few-shot 텍스트 생성
            few_shot_text = ""
            for i, example in enumerate(selected_examples, 1):
                if question_type == "multiple_choice":
                    few_shot_text += f"예시 {i}:\n문제: {example['question']}\n정답: {example['answer']}\n해설: {example['reasoning']}\n\n"
                else:
                    few_shot_text += f"예시 {i}:\n문제: {example['question']}\n답변: {example['answer']}\n\n"
            
            return few_shot_text
            
        except Exception as e:
            print(f"Few-shot 컨텍스트 구성 오류: {e}")
            return ""

    def _generate_example_signature(self, example: Dict) -> str:
        """예시 서명 생성"""
        try:
            answer = example.get("answer", "")
            question = example.get("question", "")
            return hashlib.md5(f"{question[:50]}{answer[:50]}".encode()).hexdigest()[:8]
        except Exception:
            return ""

    def build_prompt(self, question: str, question_type: str, domain: str = "일반", 
                    context_info: str = "", institution_info: str = "") -> str:
        """프롬프트 구성"""
        try:
            # Few-shot 예시 추가
            example_count = self._determine_example_count(domain, question_type, question)
            few_shot_examples = self.build_few_shot_context(domain, question_type, question, count=example_count)
            
            # 특화된 질문 유형 처리
            if self._is_ratio_question(question, domain):
                template = self.prompt_templates["ratio_specialized"]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    question=question
                )
            
            # 기관 질문 특화 처리
            if self._is_institution_question(question, domain) and institution_info:
                template = self.prompt_templates["institution_specialized"]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    institution_info=institution_info,
                    question=question
                )
            
            # 도메인 특화 템플릿 사용
            if domain in self.prompt_templates["domain_specialized"] and question_type == "subjective":
                template = self.prompt_templates["domain_specialized"][domain]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    question=question
                )
            
            # 일반 프롬프트
            if question_type == "multiple_choice":
                template = self.prompt_templates["multiple_choice_basic"]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    question=question
                )
            else:
                template = self.prompt_templates["subjective_basic"]
                context = self._enhance_context_info(context_info, domain)
                
                return template.format(
                    few_shot_examples=few_shot_examples,
                    context_info=context,
                    question=question
                )
                
        except Exception as e:
            print(f"프롬프트 구성 오류: {e}")
            return self._create_fallback_prompt(question, question_type, domain)

    def _determine_example_count(self, domain: str, question_type: str, question: str) -> int:
        """예시 개수 결정"""
        try:
            # 도메인별 기본 예시 개수
            domain_counts = {
                "사이버보안": 2,
                "전자금융": 2,
                "개인정보보호": 2,
                "정보보안": 1,
                "위험관리": 1,
                "금융투자": 1,
                "정보통신": 1
            }
            
            base_count = domain_counts.get(domain, 1)
            
            # 질문 복잡도에 따른 조정
            if len(question) > 300:
                base_count = min(base_count + 1, 2)
            elif any(word in question.lower() for word in ["특징", "지표", "방안", "절차"]):
                base_count = min(base_count + 1, 2)
            
            return base_count
        except Exception:
            return 1

    def _enhance_context_info(self, context_info: str, domain: str) -> str:
        """컨텍스트 정보 강화"""
        try:
            if not context_info:
                context_info = "관련 법령과 규정을 참고하세요."
            
            # 도메인별 추가 컨텍스트
            domain_contexts = {
                "사이버보안": "사이버보안 위협 분석 및 대응 기술 관점에서 접근하세요.",
                "전자금융": "전자금융거래법과 전자금융감독규정의 요구사항을 중심으로 고려하세요.",
                "개인정보보호": "개인정보보호법의 처리 원칙과 정보주체 권리를 중심으로 분석하세요.",
                "정보보안": "정보보안관리체계(ISMS)의 요구사항과 보안 통제 관점에서 접근하세요.",
                "위험관리": "위험관리 체계의 단계별 절차와 내부통제 관점에서 분석하세요.",
                "정보통신": "정보통신기반 보호법의 보호 요구사항을 중심으로 고려하세요."
            }
            
            if domain in domain_contexts:
                context = f"{context_info}\n\n추가 고려사항: {domain_contexts[domain]}"
                return context[:600]
            
            return context_info[:500]
            
        except Exception:
            return context_info if context_info else "관련 법령과 규정을 참고하세요."

    def _is_ratio_question(self, question: str, domain: str) -> bool:
        """비율 질문 확인"""
        question_lower = question.lower()
        
        # 강한 비율 지표
        strong_ratio_indicators = [
            "비율.*얼마", "기준.*비율.*얼마", "정보기술부문.*비율", 
            "예산.*비율", "인력.*비율", ".*%.*이상", "배정.*비율"
        ]
        
        for indicator in strong_ratio_indicators:
            if re.search(indicator, question_lower):
                return True
        
        # 전자금융 도메인의 특별 케이스
        if domain == "전자금융":
            ratio_keywords = ["비율", "기준", "정보기술부문", "예산", "인력", "배정"]
            keyword_count = sum(1 for keyword in ratio_keywords if keyword in question_lower)
            if keyword_count >= 3:
                return True
        
        return False

    def _is_institution_question(self, question: str, domain: str) -> bool:
        """기관 질문 확인"""
        question_lower = question.lower()
        
        # 강한 기관 지표
        institution_indicators = [
            "기관.*기술하세요", "기관.*설명하세요", "어떤.*기관", "어느.*기관",
            "분쟁조정.*신청.*기관", "신고.*기관", "상담.*기관", "담당.*기관"
        ]
        
        for indicator in institution_indicators:
            if re.search(indicator, question_lower):
                return True
        
        # 도메인별 기관 관련 키워드
        domain_institution_keywords = {
            "전자금융": ["분쟁조정", "전자금융분쟁조정위원회", "금융감독원"],
            "개인정보보호": ["개인정보보호위원회", "침해신고센터", "신고", "상담"],
            "정보통신": ["과학기술정보통신부"]
        }
        
        if domain in domain_institution_keywords:
            keywords = domain_institution_keywords[domain]
            if any(keyword in question_lower for keyword in keywords) and "기관" in question_lower:
                return True
        
        return False

    def _create_fallback_prompt(self, question: str, question_type: str, domain: str) -> str:
        """폴백 프롬프트 생성"""
        try:
            if question_type == "multiple_choice":
                return f"""다음은 금융보안 관련 객관식 문제입니다. 체계적 분석을 통해 정답을 선택하세요.

문제 분석:
1. 핵심 키워드 파악
2. 선택지별 검토
3. 논리적 추론 적용

문제: {question}

정답 번호: """
            else:
                return f"""다음은 금융보안 관련 주관식 문제입니다. 반드시 한국어로만 전문적이고 정확한 답변을 작성하세요.

작성 지침:
- 모든 답변은 한국어로만 작성
- 관련 법령과 규정에 근거한 답변
- 구체적이고 실무적인 내용 포함

문제: {question}

한국어 답변: """
                
        except Exception:
            return f"다음 문제에 답변하세요:\n\n{question}\n\n답변: "

    def get_context_hints(self, domain: str, intent_type: str) -> str:
        """도메인별 컨텍스트 힌트"""
        
        context_hints = {
            "사이버보안": {
                "특징_묻기": "사이버 위협의 기술적 특성과 동작 방식을 중심으로 설명하세요.",
                "지표_묻기": "네트워크 트래픽 이상, 프로세스 활동 패턴, 파일 시스템 변화 등 구체적인 탐지 지표를 포함하여 설명하세요.",
                "방안_묻기": "다층 방어체계, 실시간 모니터링, 사고 대응 절차를 포함한 종합적 대응 방안을 설명하세요."
            },
            "전자금융": {
                "기관_묻기": "전자금융거래법에 근거한 기관의 법적 지위와 구체적 업무 범위를 명확히 설명하세요.",
                "방안_묻기": "접근매체 보안, 거래 기록 보존, 분쟁조정 절차를 전자금융거래법 조항과 연계하여 설명하세요.",
                "절차_묻기": "전자금융거래법에 명시된 법적 절차와 당사자별 의무사항을 단계별로 설명하세요.",
                "비율_묻기": "전자금융감독규정 제16조에 명시된 구체적인 수치와 법적 근거를 포함하여 설명하세요."
            },
            "개인정보보호": {
                "기관_묻기": "개인정보보호법에 따른 기관의 권한과 신고 접수 절차를 구체적으로 설명하세요.",
                "방안_묻기": "수집 최소화, 목적 제한, 정보주체 권리 보장 원칙을 중심으로 종합적 방안을 설명하세요.",
                "절차_묻기": "동의 획득, 처리 현황 공개, 권리 행사 절차를 개인정보보호법 조항에 따라 설명하세요."
            },
            "정보보안": {
                "방안_묻기": "정보보안관리체계의 수립, 운영, 점검, 개선 사이클을 중심으로 설명하세요.",
                "요소_묻기": "정보보호의 3대 요소의 정의와 상호 관계를 구체적으로 설명하세요."
            },
            "위험관리": {
                "방안_묻기": "위험 식별, 평가, 대응, 모니터링의 4단계 절차를 포함하여 설명하세요."
            },
            "정보통신": {
                "방안_묻기": "정보통신기반 보호법에 따른 보고 요구사항과 절차를 명확히 설명하세요."
            }
        }
        
        try:
            return context_hints.get(domain, {}).get(intent_type, 
                "관련 법령의 구체적 조항과 실무 적용 방안을 포함하여 설명하세요.")
        except Exception:
            return "관련 법령의 구체적 조항과 실무 적용 방안을 포함하여 설명하세요."

    def cleanup(self):
        """리소스 정리"""
        try:
            self.used_examples_cache.clear()
            self.prompt_history.clear()
        except Exception as e:
            print(f"프롬프트 enhancer 정리 오류: {e}")