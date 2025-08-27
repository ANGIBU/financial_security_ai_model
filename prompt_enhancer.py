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
        self._initialize_diversity_templates()
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
                        "domain_focus": ["트로이", "특징", "위장"]
                    },
                    {
                        "question": "SBOM을 금융권에서 활용하는 가장 적절한 이유는?\n1 데이터 백업\n2 네트워크 모니터링\n3 접근 권한 관리\n4 암호화 강화\n5 소프트웨어 공급망 보안",
                        "answer": "5",
                        "reasoning": "SBOM은 소프트웨어 구성 요소의 투명성을 제공하여 공급망 보안을 강화하는 목적으로 활용됩니다.",
                        "domain_focus": ["SBOM", "활용", "공급망"]
                    },
                    {
                        "question": "딥페이크 기술의 악용을 방지하기 위한 금융권의 선제적 대응 방안으로 가장 적절한 것은?\n1 딥페이크 탐지 기능이 없는 구식 인증 시스템 도입\n2 딥보이스 탐지 기술 개발\n3 금융기관의 음성 복제\n4 딥페이크 영상 제작 지원\n5 금융소비자 홍보 강화",
                        "answer": "2",
                        "reasoning": "딥페이크 기술 악용 방지를 위한 선제적 대응 방안으로는 딥보이스 탐지 기술 개발이 가장 적절합니다.",
                        "domain_focus": ["딥페이크", "대응", "탐지"]
                    }
                ],
                "subjective": [
                    {
                        "question": "트로이 목마(Trojan) 기반 원격제어 악성코드(RAT)의 특징과 주요 탐지 지표를 설명하세요.",
                        "answer": "트로이 목마 기반 원격제어 악성코드는 정상 프로그램으로 위장하여 시스템에 침투하고 외부에서 원격으로 제어하는 특성을 가집니다. 주요 탐지 지표로는 비정상적인 네트워크 통신 패턴, 비인가 프로세스 실행, 파일 시스템 변경 등이 있으며 실시간 모니터링을 통한 종합적 분석이 필요합니다.",
                        "domain_focus": ["트로이", "특징", "탐지지표"],
                        "answer_type": "특징설명",
                        "length_category": "중간"
                    },
                    {
                        "question": "딥페이크 기술 악용에 대비한 금융권의 대응 방안을 기술하세요.",
                        "answer": "딥페이크 기술 악용에 대비하여 다층 방어체계 구축, 실시간 탐지 시스템 도입, 생체인증과 다중 인증 체계를 통한 신원 검증 강화, 직원 교육 및 인식 제고를 통한 종합적 보안 대응방안이 필요합니다.",
                        "domain_focus": ["딥페이크", "대응", "방안"],
                        "answer_type": "방안제시",
                        "length_category": "중간"
                    },
                    {
                        "question": "디지털 지갑(Digital Wallet)에서 우려되는 주요 보안 위협을 설명하세요.",
                        "answer": "디지털 지갑의 주요 보안 위협으로는 개인키 도난, 피싱 공격, 멀웨어 감염, 스마트 컨트랙트 취약점이 있으며 다중 인증과 하드웨어 지갑 사용이 권장됩니다.",
                        "domain_focus": ["디지털지갑", "보안위협"],
                        "answer_type": "위협설명",
                        "length_category": "짧음"
                    }
                ]
            },
            "전자금융": {
                "multiple_choice": [
                    {
                        "question": "한국은행이 금융통화위원회의 요청에 따라 전자금융업자에게 자료제출을 요구할 수 있는 경우로 가장 적절한 것은?\n1 보안 시스템 점검\n2 고객 정보 확인\n3 경영 실적 조사\n4 통화신용정책 수행\n5 시장 동향 파악",
                        "answer": "4",
                        "reasoning": "한국은행법에 따라 통화신용정책의 수행 및 지급결제제도의 원활한 운영을 위해 자료제출을 요구할 수 있습니다.",
                        "domain_focus": ["한국은행", "자료제출", "통화신용정책"]
                    }
                ],
                "subjective": [
                    {
                        "question": "전자금융거래법에 따라 이용자가 금융 분쟁조정을 신청할 수 있는 기관을 기술하세요.",
                        "answer": "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당하며, 금융감독원 내에 설치되어 전자금융거래법에 근거하여 이용자와 전자금융업자 간의 분쟁을 공정하고 신속하게 해결합니다.",
                        "domain_focus": ["분쟁조정", "기관", "전자금융거래법"],
                        "answer_type": "기관명",
                        "length_category": "중간"
                    },
                    {
                        "question": "금융회사가 정보보호 예산을 관리할 때, 전자금융감독규정상 정보기술부문 인력 및 예산의 기준 비율은 얼마인가요?",
                        "answer": "전자금융감독규정 제16조에 따라 금융회사는 정보기술부문 인력을 총 인력의 5% 이상, 정보기술부문 예산을 총 예산의 7% 이상 정보보호 업무에 배정해야 합니다. 다만 회사 규모, 업무 특성, 정보기술 위험수준 등에 따라 금융감독원장이 별도로 정할 수 있습니다.",
                        "domain_focus": ["정보기술부문", "비율", "예산"],
                        "answer_type": "수치설명",
                        "length_category": "긴"
                    },
                    {
                        "question": "전자금융업자가 수행해야 할 보안조치의 주요 내용을 설명하세요.",
                        "answer": "전자금융업자는 이용자의 전자금융거래 안전성 확보를 위한 보안조치를 시행하고 접근매체의 안전한 보관 및 관리, 거래기록의 보존과 위조변조 방지, 암호화 기술을 통한 거래정보 보호 등 종합적인 보안체계를 구축해야 합니다.",
                        "domain_focus": ["보안조치", "전자금융업자"],
                        "answer_type": "조치설명",
                        "length_category": "중간"
                    }
                ]
            },
            "개인정보보호": {
                "multiple_choice": [
                    {
                        "question": "만 14세 미만 아동의 개인정보 처리를 위해 필요한 절차로 가장 적절한 것은?\n1 본인의 직접 동의\n2 법정대리인의 동의\n3 학교의 승인\n4 관할 기관 허가\n5 보호자 확인서",
                        "answer": "2",
                        "reasoning": "개인정보보호법에 따라 만 14세 미만 아동의 개인정보 처리에는 법정대리인의 동의가 필요합니다.",
                        "domain_focus": ["만14세", "아동", "법정대리인"]
                    }
                ],
                "subjective": [
                    {
                        "question": "개인정보 접근 권한 검토는 어떻게 수행해야 하며, 그 목적은 무엇인가요?",
                        "answer": "개인정보 접근 권한 검토는 업무상 필요한 최소한의 권한만을 부여하는 최소권한 원칙에 따라 정기적으로 수행하며, 불필요한 권한은 즉시 회수하여 개인정보 오남용을 방지해야 합니다.",
                        "domain_focus": ["접근권한", "검토", "최소권한"],
                        "answer_type": "절차설명",
                        "length_category": "중간"
                    },
                    {
                        "question": "개인정보 관리체계 수립 및 운영의 정책 수립 단계에서 가장 중요한 요소를 설명하세요.",
                        "answer": "개인정보 관리체계의 정책 수립 단계에서 가장 중요한 요소는 경영진의 적극적인 참여와 의지입니다. 최고 경영진의 개인정보보호에 대한 확고한 의지와 지원이 있어야 체계적이고 효과적인 관리체계를 구축할 수 있습니다.",
                        "domain_focus": ["관리체계", "정책수립", "경영진"],
                        "answer_type": "요소설명",
                        "length_category": "중간"
                    }
                ]
            },
            "정보보안": {
                "multiple_choice": [
                    {
                        "question": "재해 복구 계획 수립 시 고려 요소 중 옳지 않은 것은?\n1 복구 절차 수립\n2 비상연락체계 구축\n3 개인정보 파기 절차\n4 복구 목표시간 설정\n5 백업 시스템 구축",
                        "answer": "3",
                        "reasoning": "개인정보 파기 절차는 재해 복구 계획과 직접적인 관련이 없으며, 복구 관련 요소가 아닙니다.",
                        "domain_focus": ["재해복구", "계획수립", "파기절차"]
                    }
                ],
                "subjective": [
                    {
                        "question": "정보보호의 3대 요소에 해당하는 보안 목표를 3가지 기술하세요.",
                        "answer": "정보보호의 3대 요소는 기밀성(Confidentiality), 무결성(Integrity), 가용성(Availability)으로 구성되며, 이를 통해 정보자산의 안전한 보호와 관리를 보장합니다.",
                        "domain_focus": ["3대요소", "보안목표"],
                        "answer_type": "정의설명",
                        "length_category": "중간"
                    },
                    {
                        "question": "SMTP 프로토콜의 보안상 주요 역할을 설명하세요.",
                        "answer": "SMTP 프로토콜은 이메일 전송을 담당하며, 보안상 주요 역할로는 인증 메커니즘 제공, 암호화 통신 지원, 스팸 및 악성 이메일 차단을 통해 안전한 이메일 서비스를 보장합니다.",
                        "domain_focus": ["SMTP", "프로토콜", "보안역할"],
                        "answer_type": "역할설명",
                        "length_category": "중간"
                    }
                ]
            },
            "정보통신": {
                "multiple_choice": [
                    {
                        "question": "집적된 정보통신시설의 보호와 관련하여 정보통신서비스 제공의 중단이 발생했을 때, 과학기술정보통신부장관에게 보고해야 하는 사항으로 옳지 않은 것은?\n1 정보통신서비스 제공의 중단이 발생한 일시 및 장소\n2 정보통신서비스 제공의 중단이 발생한 원인에 대한 법적 책임\n3 정보통신서비스 제공의 중단이 발생한 원인 및 피해내용\n4 응급조치 사항",
                        "answer": "2",
                        "reasoning": "정보통신서비스 제공 중단 발생 시 과학기술정보통신부장관에게 보고해야 하는 사항에는 법적 책임이 포함되지 않습니다.",
                        "domain_focus": ["정보통신서비스", "중단", "보고사항"]
                    }
                ]
            },
            "금융투자": {
                "multiple_choice": [
                    {
                        "question": "금융산업의 이해와 관련하여 금융투자업의 구분에 해당하지 않는 것은?\n1 소비자금융업\n2 투자자문업\n3 투자매매업\n4 투자중개업\n5 보험중개업",
                        "answer": "1",
                        "reasoning": "소비자금융업은 금융투자업에 해당하지 않으며, 별도의 금융업 분류에 속합니다.",
                        "domain_focus": ["금융투자업", "구분", "소비자금융업"]
                    }
                ]
            },
            "위험관리": {
                "multiple_choice": [
                    {
                        "question": "위험 관리 계획 수립 시 고려해야 할 요소로 적절하지 않은 것은?\n1 수행인력\n2 위험 수용\n3 위험 대응 전략 선정\n4 대상\n5 기간",
                        "answer": "2",
                        "reasoning": "위험 관리 계획에서 위험 수용은 적절한 관리 요소가 아니며, 위험을 식별하고 대응하는 것이 중요합니다.",
                        "domain_focus": ["위험관리", "계획수립", "위험수용"]
                    }
                ]
            }
        }
    
    def _initialize_prompt_templates(self):
        """프롬프트 템플릿 초기화"""
        
        self.prompt_templates = {
            "multiple_choice_base": """다음은 금융보안 관련 객관식 문제입니다. 주어진 선택지 중에서 가장 적절한 답을 선택하세요.

{few_shot_examples}

문제: {question}

위 문제를 다음 단계로 분석하여 정답을 선택하세요:
1. 문제의 핵심 키워드와 요구사항 파악
2. 각 선택지를 해당 법령과 규정에 따라 검토
3. 문제 유형(부정/긍정)에 따른 논리적 추론
4. 전문가 관점에서 최적의 답안 선택

정답 번호: """,

            "subjective_base": """다음은 금융보안 관련 주관식 문제입니다. 반드시 한국어로만 전문적이고 정확한 답변을 작성하세요.

답변 작성 지침:
- 모든 답변은 한국어로만 작성
- 관련 법령과 규정에 근거한 전문적 답변 작성
- 구체적이고 실무적인 내용 포함
- 자연스러운 한국어 문장으로 구성
{diversity_instruction}

{few_shot_examples}

참고 정보:
{context_info}

문제: {question}

위 문제에 대해 관련 법령과 규정을 근거로 구체적이고 전문적인 한국어 답변을 작성하세요.

한국어 답변: """,

            "institution_question": """다음은 금융보안 관련 기관에 대한 질문입니다. 반드시 한국어로만 답변하세요.

답변 작성 지침:
- 모든 답변은 한국어로만 작성
- 기관의 정확한 명칭과 역할 기술

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

            "ratio_question": """다음은 금융보안 관련 비율에 대한 질문입니다. 반드시 구체적인 수치와 법적 근거를 포함하여 한국어로만 답변하세요.

답변 작성 지침:
- 정확한 수치와 퍼센트 명시
- 해당 법령과 조항 인용
- 예외 조건이나 특별 규정 포함

{few_shot_examples}

문제: {question}

위 질문에 대해 다음 사항을 포함하여 한국어로 답변하세요:
1. 정확한 비율과 수치
2. 관련 법령과 조항 번호
3. 적용 조건 및 예외사항
4. 감독기관의 재량권

한국어 답변: """,

            "domain_specific": {
                "사이버보안": """다음은 사이버보안 관련 문제입니다. 기술적 특성과 보안 대응 방안에 중점을 두어 답변하세요.

{few_shot_examples}

문제: {question}

사이버보안 관점에서 기술적 특징, 탐지 방법, 대응 방안을 구체적으로 설명하세요.

한국어 답변: """,

                "전자금융": """다음은 전자금융 관련 문제입니다. 전자금융거래법과 관련 규정을 근거로 답변하세요.

{few_shot_examples}

문제: {question}

전자금융거래법에 근거하여 법적 요구사항과 절차를 명확히 설명하세요.

한국어 답변: """,

                "개인정보보호": """다음은 개인정보보호 관련 문제입니다. 개인정보보호법을 근거로 답변하세요.

{few_shot_examples}

문제: {question}

개인정보보호법에 따른 처리 원칙과 절차를 구체적으로 설명하세요.

한국어 답변: """,

                "정보보안": """다음은 정보보안 관련 문제입니다. 정보보안관리체계 관점에서 답변하세요.

{few_shot_examples}

문제: {question}

정보보안관리체계 구축과 운영 관점에서 체계적으로 설명하세요.

한국어 답변: """,

                "정보통신": """다음은 정보통신 관련 문제입니다. 정보통신기반 보호법을 근거로 답변하세요.

{few_shot_examples}

문제: {question}

정보통신기반 보호법에 따른 요구사항과 절차를 명확히 설명하세요.

한국어 답변: """
            }
        }

    def _initialize_diversity_templates(self):
        """다양성 확보 템플릿 초기화"""
        
        self.diversity_instructions = {
            "legal_focus": "\n- 법적 조항과 규정에 중점을 두어 답변하세요.",
            "practical_focus": "\n- 실무적이고 구체적인 절차를 중심으로 답변하세요.", 
            "technical_focus": "\n- 기술적 특성과 메커니즘을 상세히 설명하세요.",
            "process_focus": "\n- 단계별 과정과 절차를 명확히 제시하세요.",
            "comprehensive_focus": "\n- 종합적이고 포괄적인 관점에서 답변하세요."
        }
        
        self.context_variations = {
            "formal": "법령과 규정에 근거한 공식적인 관점에서",
            "practical": "실무 적용과 현장 경험을 바탕으로",
            "systematic": "체계적이고 논리적인 접근을 통해", 
            "comprehensive": "다각적이고 종합적인 분석을 통해",
            "specific": "구체적인 사례와 절차를 중심으로"
        }

    def _generate_prompt_hash(self, question: str, domain: str, question_type: str) -> str:
        """프롬프트 해시 생성"""
        try:
            combined_text = f"{question[:50]}-{domain}-{question_type}"
            return hashlib.md5(combined_text.encode()).hexdigest()[:8]
        except Exception:
            return ""

    def _select_diverse_examples(self, domain: str, question_type: str, question: str, count: int = 2) -> List[Dict]:
        """다양성을 고려한 예시 선택"""
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
            
            example_scores = []
            for example in examples:
                score = 0
                
                # 도메인 포커스 키워드 매칭
                if "domain_focus" in example:
                    focus_matches = sum(1 for focus in example["domain_focus"] 
                                      if focus.lower() in question.lower())
                    score += focus_matches * 2
                
                # 답변 유형 매칭 (주관식만)
                if question_type == "subjective" and "answer_type" in example:
                    if self._match_answer_type(question, example["answer_type"]):
                        score += 3
                
                # 길이 다양성 고려
                if "length_category" in example:
                    length_bonus = {"짧음": 1, "중간": 2, "긴": 3}
                    score += length_bonus.get(example["length_category"], 1)
                
                example_scores.append((example, score))
            
            # 점수 기준 정렬
            example_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 다양성 고려하여 선택
            selected_examples = []
            used_answer_types = set()
            
            for example, score in example_scores:
                if len(selected_examples) >= count:
                    break
                
                answer_type = example.get("answer_type", "default")
                if question_type == "subjective" and answer_type in used_answer_types:
                    continue
                
                selected_examples.append(example)
                if question_type == "subjective":
                    used_answer_types.add(answer_type)
            
            # 부족한 경우 추가 선택
            if len(selected_examples) < count:
                remaining = [ex for ex, _ in example_scores if ex not in selected_examples]
                selected_examples.extend(remaining[:count - len(selected_examples)])
            
            return selected_examples[:count]
            
        except Exception as e:
            print(f"예시 선택 오류: {e}")
            return examples[:count] if examples else []

    def _match_answer_type(self, question: str, answer_type: str) -> bool:
        """답변 유형 매칭 확인"""
        question_lower = question.lower()
        
        type_patterns = {
            "기관명": ["기관", "위원회", "담당", "어디"],
            "특징설명": ["특징", "특성", "성질", "어떤"],
            "지표나열": ["지표", "징후", "탐지", "모니터링"],
            "방안제시": ["방안", "대책", "대응", "해결"],
            "절차설명": ["절차", "과정", "단계", "어떻게"],
            "수치설명": ["비율", "얼마", "기준", "퍼센트"],
            "정의설명": ["정의", "무엇", "개념", "의미"],
            "역할설명": ["역할", "기능", "업무", "담당"]
        }
        
        if answer_type in type_patterns:
            return any(pattern in question_lower for pattern in type_patterns[answer_type])
        
        return False

    def build_few_shot_context(self, domain: str, question_type: str, question: str, count: int = 2) -> str:
        """Few-shot 예시 구성"""
        try:
            # 프롬프트 해시 생성
            prompt_hash = self._generate_prompt_hash(question, domain, question_type)
            
            # 이전에 사용된 예시 확인
            if prompt_hash in self.used_examples_cache:
                # 다른 예시 선택을 위해 제외 목록 활용
                excluded_examples = self.used_examples_cache[prompt_hash]
            else:
                excluded_examples = []
            
            if question_type == "subjective":
                count = min(count, 1)  # 주관식은 1개만
            
            selected_examples = self._select_diverse_examples(domain, question_type, question, count)
            
            # 제외 목록에서 다른 예시 찾기
            if excluded_examples:
                available_examples = [ex for ex in selected_examples 
                                    if ex.get("answer", "")[:30] not in excluded_examples]
                if available_examples:
                    selected_examples = available_examples
            
            # 사용된 예시 캐시 업데이트
            if selected_examples:
                used_answers = [ex.get("answer", "")[:30] for ex in selected_examples]
                self.used_examples_cache[prompt_hash] = used_answers
                
                # 캐시 크기 제한
                if len(self.used_examples_cache) > 50:
                    oldest_key = list(self.used_examples_cache.keys())[0]
                    del self.used_examples_cache[oldest_key]
            
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

    def _get_diversity_instruction(self, domain: str, force_diversity: bool = False) -> str:
        """다양성 지침 선택"""
        if not force_diversity:
            return ""
        
        try:
            # 도메인별 선호 지침
            domain_preferences = {
                "사이버보안": ["technical_focus", "process_focus"],
                "전자금융": ["legal_focus", "practical_focus"],
                "개인정보보호": ["legal_focus", "process_focus"],
                "정보보안": ["systematic_focus", "technical_focus"],
                "위험관리": ["systematic_focus", "process_focus"],
                "금융투자": ["legal_focus", "practical_focus"],
                "정보통신": ["legal_focus", "process_focus"]
            }
            
            available_instructions = domain_preferences.get(domain, list(self.diversity_instructions.keys()))
            selected_instruction = random.choice(available_instructions)
            
            return self.diversity_instructions.get(selected_instruction, "")
            
        except Exception:
            return ""

    def build_enhanced_prompt(self, question: str, question_type: str, domain: str = "일반", 
                            context_info: str = "", institution_info: str = "", 
                            force_diversity: bool = False) -> str:
        """프롬프트 구성"""
        try:
            # Few-shot 예시 추가
            example_count = 2 if domain in ["개인정보보호", "전자금융"] else 1
            few_shot_examples = self.build_few_shot_context(domain, question_type, question, count=example_count)
            
            # 다양성 지침 생성
            diversity_instruction = self._get_diversity_instruction(domain, force_diversity)
            
            # 비율 관련 질문 특별 처리
            if self._is_ratio_question(question, domain):
                template = self.prompt_templates["ratio_question"]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    question=question
                )
            
            # 기관 질문 특별 처리
            if ("기관" in question.lower() or "위원회" in question.lower()) and institution_info:
                template = self.prompt_templates["institution_question"]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    institution_info=institution_info,
                    question=question
                )
            
            # 도메인별 특화 템플릿 사용
            if domain in self.prompt_templates["domain_specific"] and question_type == "subjective":
                template = self.prompt_templates["domain_specific"][domain]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    question=question
                )
            
            # 일반 프롬프트
            if question_type == "multiple_choice":
                template = self.prompt_templates["multiple_choice_base"]
                return template.format(
                    few_shot_examples=few_shot_examples,
                    question=question
                )
            else:
                template = self.prompt_templates["subjective_base"]
                simplified_context = context_info[:500] + "..." if len(context_info) > 500 else context_info
                
                return template.format(
                    few_shot_examples=few_shot_examples,
                    context_info=simplified_context if simplified_context else "관련 법령과 규정을 참고하세요.",
                    question=question,
                    diversity_instruction=diversity_instruction
                )
                
        except Exception as e:
            print(f"프롬프트 구성 오류: {e}")
            if question_type == "multiple_choice":
                return f"""다음 문제의 정답 번호를 선택하세요.

문제: {question}

정답 번호: """
            else:
                diversity_note = ""
                if force_diversity:
                    diversity_note = "\n\n중요: 이전과 다른 구체적이고 실무적인 관점에서 답변하세요."
                    
                return f"""다음 문제에 대해 한국어로만 전문적인 답변을 작성하세요.{diversity_note}

문제: {question}

한국어 답변: """

    def _is_ratio_question(self, question: str, domain: str) -> bool:
        """비율 관련 질문 확인"""
        question_lower = question.lower()
        
        ratio_indicators = [
            "비율", "얼마", "기준", "퍼센트", "%", 
            "정보기술부문", "인력", "예산", "배정"
        ]
        
        # 전자금융 도메인에서 정보기술부문 관련 질문
        if domain == "전자금융":
            if any(indicator in question_lower for indicator in ratio_indicators):
                if "정보기술부문" in question_lower or "예산" in question_lower:
                    return True
        
        # 일반적인 비율 질문
        ratio_count = sum(1 for indicator in ratio_indicators if indicator in question_lower)
        return ratio_count >= 2
    
    def get_context_hints(self, domain: str, intent_type: str) -> str:
        """도메인별 컨텍스트 힌트 제공"""
        
        context_hints = {
            "사이버보안": {
                "특징_묻기": "사이버 위협의 기술적 특성과 동작 방식, 은밀성과 지속성을 중심으로 설명하세요.",
                "지표_묻기": "네트워크 트래픽, 프로세스 활동, 파일 시스템 변화 등 구체적인 탐지 지표를 포함하여 설명하세요.",
                "방안_묻기": "다층 방어체계, 실시간 모니터링, 사고 대응 절차를 포함한 종합적 방안을 설명하세요."
            },
            "전자금융": {
                "기관_묻기": "전자금융거래법에 근거한 기관의 법적 지위와 구체적 업무 범위를 명확히 설명하세요.",
                "방안_묻기": "접근매체 보안, 거래 기록 보존, 분쟁조정 절차를 포함한 이용자 보호 방안을 설명하세요.",
                "절차_묻기": "전자금융거래법에 명시된 법적 절차와 당사자별 의무사항을 단계별로 설명하세요.",
                "비율_묻기": "전자금융감독규정에 명시된 구체적인 수치와 법적 근거를 포함하여 설명하세요."
            },
            "개인정보보호": {
                "기관_묻기": "개인정보보호법에 따른 기관의 권한과 개인정보 처리 감독 업무를 구체적으로 설명하세요.",
                "방안_묻기": "수집 최소화, 목적 제한, 정보주체 권리 보장 원칙을 중심으로 설명하세요.",
                "절차_묻기": "동의 획득, 처리 현황 공개, 권리 행사 절차를 법령에 따라 설명하세요."
            },
            "금융투자": {
                "방안_묻기": "자본시장법의 투자자 보호 원칙과 적합성 원칙 적용 방안을 중심으로 설명하세요."
            },
            "위험관리": {
                "방안_묻기": "위험 식별, 평가, 대응, 모니터링의 4단계 절차와 각 단계별 핵심 활동을 설명하세요."
            },
            "정보보안": {
                "방안_묻기": "정보보안관리체계의 수립, 운영, 점검, 개선 사이클을 중심으로 설명하세요."
            },
            "정보통신": {
                "방안_묻기": "정보통신기반 보호법에 따른 보고 요구사항과 절차를 명확히 설명하세요."
            }
        }
        
        try:
            return context_hints.get(domain, {}).get(intent_type, "관련 법령의 구체적 조항과 실무 적용 방안을 포함하여 체계적으로 설명하세요.")
        except Exception:
            return "관련 법령의 구체적 조항과 실무 적용 방안을 포함하여 체계적으로 설명하세요."
    
    def analyze_prompt_effectiveness(self, question: str, answer: str, domain: str) -> Dict:
        """프롬프트 효과성 분석"""
        try:
            analysis = {
                "answer_length": len(answer) if answer else 0,
                "korean_ratio": self._calculate_korean_ratio(answer) if answer else 0,
                "domain_relevance": self._check_domain_relevance(answer, domain) if answer else 0,
                "uniqueness_score": self._calculate_uniqueness(answer) if answer else 0,
                "quality_score": 0
            }
            
            # 품질 점수 계산
            if answer:
                quality_factors = [
                    analysis["korean_ratio"] > 0.8,
                    analysis["answer_length"] > 50,
                    analysis["domain_relevance"] > 0.7,
                    analysis["uniqueness_score"] > 0.6
                ]
                analysis["quality_score"] = sum(quality_factors) / len(quality_factors)
            
            return analysis
            
        except Exception as e:
            print(f"프롬프트 효과성 분석 오류: {e}")
            return {"quality_score": 0.5}

    def _calculate_korean_ratio(self, text: str) -> float:
        """한국어 비율 계산"""
        if not text:
            return 0.0
        
        try:
            korean_chars = len(re.findall(r"[가-힣]", text))
            total_chars = len(re.sub(r"[^\w가-힣]", "", text))
            return korean_chars / total_chars if total_chars > 0 else 0.0
        except Exception:
            return 0.0

    def _check_domain_relevance(self, text: str, domain: str) -> float:
        """도메인 관련성 확인"""
        if not text or not domain:
            return 0.0
        
        try:
            domain_keywords = {
                "사이버보안": ["보안", "악성코드", "탐지", "방어", "위협"],
                "전자금융": ["전자금융", "거래", "분쟁", "조정", "보안"],
                "개인정보보호": ["개인정보", "정보주체", "동의", "처리", "보호"],
                "정보보안": ["정보보안", "관리체계", "접근통제", "암호화"],
                "위험관리": ["위험", "관리", "평가", "대응", "모니터링"],
                "금융투자": ["투자", "금융", "자본시장", "투자자", "보호"],
                "정보통신": ["정보통신", "시설", "서비스", "보고", "중단"]
            }
            
            keywords = domain_keywords.get(domain, [])
            if not keywords:
                return 0.5
                
            text_lower = text.lower()
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            
            return min(matches / len(keywords), 1.0)
            
        except Exception:
            return 0.5

    def _calculate_uniqueness(self, text: str) -> float:
        """답변 고유성 계산"""
        if not text:
            return 0.0
        
        try:
            # 간단한 고유성 측정 (반복 패턴 감지)
            words = text.split()
            unique_words = set(words)
            
            if len(words) == 0:
                return 0.0
                
            uniqueness_ratio = len(unique_words) / len(words)
            
            # 일반적인 단어 패턴 감지
            common_patterns = ["에 따라", "해야 합니다", "필요합니다", "관련", "관리"]
            pattern_count = sum(1 for pattern in common_patterns if pattern in text)
            pattern_penalty = min(pattern_count * 0.1, 0.3)
            
            return max(uniqueness_ratio - pattern_penalty, 0.0)
            
        except Exception:
            return 0.5
    
    def cleanup(self):
        """리소스 정리"""
        self.used_examples_cache.clear()
        self.prompt_history.clear()