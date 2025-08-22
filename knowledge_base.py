# knowledge_base.py

"""
금융보안 지식베이스 - 템플릿 활용 대폭 강화
- 스마트 템플릿 선택 및 융합
- 자연스러운 답변 구조 제공
- 의도별 맞춤형 지식 제공
- 도메인 전문성 강화
- 고품질 답변 생성 지원
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import random

# 설정 파일 import
from config import JSON_CONFIG_FILES, TEMPLATE_QUALITY_CRITERIA


class FinancialSecurityKnowledgeBase:
    """금융보안 지식베이스 - 템플릿 활용 및 답변 생성 특화"""

    def __init__(self):
        # JSON 설정 파일 로드
        self._load_json_configs()

        # 템플릿 품질 평가 기준
        self.template_quality_criteria = TEMPLATE_QUALITY_CRITERIA

        # 고급 템플릿 활용 설정
        self.advanced_template_config = {
            "smart_selection": True,        # 스마트 선택
            "template_fusion": True,        # 템플릿 융합
            "natural_adaptation": True,     # 자연스러운 적응
            "quality_filtering": True,      # 품질 필터링
            "intent_optimization": True,    # 의도 최적화
        }

        # 템플릿 사용 통계 강화
        self.template_usage_stats = {
            "total_requests": 0,
            "template_provided": 0,
            "smart_selection_used": 0,
            "template_fusion_used": 0,
            "natural_adaptation_used": 0,
            "fallback_used": 0,
            "domain_distribution": {},
            "intent_distribution": {},
            "quality_scores": [],
        }

        # 답변 품질 향상을 위한 구조 패턴
        self.answer_structure_patterns = {
            "기관_묻기": {
                "opening": ["에서", "가", "는"],
                "content": ["담당하며", "수행하며", "관리하며"],
                "closing": ["있습니다", "됩니다", "수행합니다"],
                "connectors": ["그리고", "또한", "아울러"],
                "professional_terms": ["업무", "역할", "기능", "담당"]
            },
            "특징_묻기": {
                "opening": ["주요 특징은", "핵심 특성은", "주된 성질은"],
                "content": ["특성을 가지며", "특징을 보이며", "성질을 나타내며"],
                "closing": ["특징입니다", "특성입니다", "성질입니다"],
                "connectors": ["특히", "더불어", "또한"],
                "professional_terms": ["특징", "특성", "성질", "원리"]
            },
            "지표_묻기": {
                "opening": ["주요 탐지 지표로는", "핵심 지표는", "주된 징후는"],
                "content": ["통해 탐지하며", "으로 식별하며", "를 통해 확인하며"],
                "closing": ["탐지됩니다", "식별됩니다", "확인됩니다"],
                "connectors": ["그리고", "더불어", "아울러"],
                "professional_terms": ["지표", "징후", "패턴", "탐지"]
            },
            "방안_묻기": {
                "opening": ["효과적인 방안으로는", "주요 대책은", "적절한 조치는"],
                "content": ["을 통해", "를 활용하여", "을 구축하여"],
                "closing": ["필요합니다", "중요합니다", "해야 합니다"],
                "connectors": ["또한", "더불어", "아울러"],
                "professional_terms": ["방안", "대책", "조치", "관리"]
            }
        }

    def _load_json_configs(self):
        """JSON 설정 파일 로드"""
        try:
            # knowledge_data.json 로드
            with open(JSON_CONFIG_FILES["knowledge_data"], "r", encoding="utf-8") as f:
                knowledge_data = json.load(f)

            # 지식베이스 데이터 할당
            self.korean_subjective_templates = knowledge_data[
                "korean_subjective_templates"
            ]
            self.domain_keywords = knowledge_data["domain_keywords"]
            self.korean_financial_terms = knowledge_data["korean_financial_terms"]
            self.institution_database = knowledge_data["institution_database"]
            self.mc_answer_patterns = knowledge_data["mc_answer_patterns"]

            # 템플릿 품질 향상 처리
            self._enhance_template_quality()

            print("지식베이스 설정 파일 로드 완료 - 템플릿 활용 강화")

        except FileNotFoundError as e:
            print(f"설정 파일을 찾을 수 없습니다: {e}")
            self._load_default_configs()
        except json.JSONDecodeError as e:
            print(f"JSON 파일 파싱 오류: {e}")
            self._load_default_configs()
        except Exception as e:
            print(f"설정 파일 로드 중 오류: {e}")
            self._load_default_configs()

    def _enhance_template_quality(self):
        """템플릿 품질 향상 처리"""
        
        # 기존 템플릿들의 품질 점수 계산 및 정렬
        for domain, domain_templates in self.korean_subjective_templates.items():
            if isinstance(domain_templates, dict):
                for intent, templates in domain_templates.items():
                    if isinstance(templates, list):
                        # 템플릿 품질 점수 계산 및 정렬
                        quality_scored_templates = []
                        for template in templates:
                            quality_score = self._calculate_template_quality_score(template)
                            quality_scored_templates.append((template, quality_score))
                        
                        # 품질 점수순으로 정렬
                        quality_scored_templates.sort(key=lambda x: x[1], reverse=True)
                        
                        # 고품질 템플릿만 유지 (점수 0.6 이상)
                        high_quality_templates = [
                            template for template, score in quality_scored_templates 
                            if score >= 0.6
                        ]
                        
                        if high_quality_templates:
                            domain_templates[intent] = high_quality_templates
                        else:
                            # 모든 템플릿이 기준 미달이면 상위 3개 유지
                            domain_templates[intent] = [
                                template for template, score in quality_scored_templates[:3]
                            ]

    def _calculate_template_quality_score(self, template: str) -> float:
        """템플릿 품질 점수 계산"""
        if not template:
            return 0.0
        
        score = 0.0
        
        # 길이 적절성 (30-300자)
        length = len(template)
        if 30 <= length <= 300:
            score += 0.25
        elif 20 <= length < 30 or 300 < length <= 400:
            score += 0.15
        
        # 한국어 비율
        korean_chars = len([c for c in template if '\uAC00' <= c <= '\uD7A3'])
        total_chars = len([c for c in template if c.isalpha()])
        if total_chars > 0:
            korean_ratio = korean_chars / total_chars
            score += min(korean_ratio, 1.0) * 0.25
        
        # 전문 용어 포함
        professional_terms = [
            "법령", "규정", "조치", "관리", "보안", "방안", "절차", "기준",
            "정책", "체계", "시스템", "위원회", "기관", "담당", "업무"
        ]
        term_count = sum(1 for term in professional_terms if term in template)
        score += min(term_count / 5, 1.0) * 0.2
        
        # 문장 완성도
        if template.endswith(("다.", "요.", "함.", "니다.", "습니다.")):
            score += 0.15
        
        # 자연스러운 연결어 포함
        natural_connectors = ["또한", "더불어", "아울러", "그리고", "이를", "따라서"]
        if any(connector in template for connector in natural_connectors):
            score += 0.1
        
        # 반복 패턴 페널티
        if self._has_repetitive_pattern(template):
            score -= 0.3
        
        return min(score, 1.0)

    def _has_repetitive_pattern(self, text: str) -> bool:
        """반복 패턴 감지"""
        if not text or len(text) < 20:
            return False
        
        # 단어 반복 감지
        words = text.split()
        for i, word in enumerate(words):
            if len(word) > 2:
                count = words.count(word)
                if count >= 3:  # 같은 단어가 3번 이상
                    return True
        
        # 구문 반복 감지
        phrases = re.findall(r'[^.,!?]+[.,!?]', text)
        unique_phrases = set(phrase.strip() for phrase in phrases if len(phrase.strip()) > 5)
        if len(phrases) > 2 and len(unique_phrases) < len(phrases) * 0.7:
            return True
        
        return False

    def _load_default_configs(self):
        """기본 설정 로드"""
        print("기본 설정으로 대체합니다.")

        # 향상된 기본 템플릿들
        self.korean_subjective_templates = {
            "사이버보안": {
                "특징_묻기": [
                    "트로이 목마 기반 원격제어 악성코드는 정상적인 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가지며, 설치 후 외부 공격자가 원격으로 시스템을 제어할 수 있는 백도어를 생성하여 은밀성과 지속성을 특징으로 합니다.",
                    "해당 악성코드는 사용자를 속여 시스템에 침투하는 사회공학적 특성과 함께 시스템 깊숙이 숨어서 장기간 활동하면서 정보 수집과 원격 제어 기능을 수행하는 특징을 가집니다.",
                    "주요 특징으로는 정상 소프트웨어로 가장하는 위장성, 사용자의 자발적 설치를 유도하는 기만성, 설치 후 시스템 권한을 탈취하는 침투성, 외부와의 은밀한 통신을 수행하는 원격제어성이 있습니다."
                ],
                "지표_묻기": [
                    "주요 탐지 지표로는 네트워크 트래픽 모니터링에서 비정상적인 외부 통신 패턴, 시스템 동작 분석에서 비인가 프로세스 실행, 파일 시스템 변경 감시에서 이상 패턴, 그리고 시스템 성능 저하 현상을 종합적으로 분석하여 탐지할 수 있습니다.",
                    "네트워크 연결 상태의 이상 징후, 예상치 못한 아웃바운드 트래픽, 시스템 리소스 과다 사용, 파일 무결성 변경, 레지스트리 수정 흔적 등이 주요 탐지 지표로 활용됩니다.",
                    "실시간 모니터링을 통한 비정상 프로세스 탐지, 방화벽 로그 분석을 통한 의심스러운 연결 식별, 시스템 콜 추적을 통한 악의적 행위 감지 등의 지표를 활용할 수 있습니다."
                ],
                "방안_묻기": [
                    "효과적인 대응방안으로는 다층 방어체계 구축을 통한 예방 조치, 실시간 모니터링 시스템 도입을 통한 조기 탐지, 사고 대응 절차 수립을 통한 신속한 대응, 그리고 정기적인 보안 교육을 통한 인식 개선이 필요합니다.",
                    "네트워크 분할을 통한 피해 확산 방지, 접근권한 최소화 원칙 적용, 행위 기반 탐지 시스템 구축, 백업 및 복구 체계 마련 등의 종합적 보안 강화 방안을 수립해야 합니다."
                ]
            },
            "개인정보보호": {
                "기관_묻기": [
                    "개인정보보호위원회에서 개인정보 보호에 관한 업무를 총괄하며, 개인정보침해신고센터에서 개인정보 침해신고 접수 및 상담 업무를 담당하고 있습니다.",
                    "개인정보 보호 정책 수립과 감시 업무는 개인정보보호위원회에서 담당하며, 개인정보 분쟁조정위원회에서 관련 분쟁의 조정 업무를 수행합니다."
                ],
                "방안_묻기": [
                    "개인정보보호법에 따라 수집 최소화 원칙을 적용하고 목적 외 이용을 금지하며, 적절한 기술적·관리적·물리적 보호조치를 수립하여 정보주체의 권리를 보장하는 종합적 관리방안이 필요합니다.",
                    "개인정보 처리 전 과정에서 개인정보처리방침 수립 및 공개, 개인정보보호책임자 지정, 정기적인 개인정보 영향평가 실시, 직원 교육 강화 등의 체계적 관리 체계를 구축해야 합니다."
                ]
            },
            "전자금융": {
                "기관_묻기": [
                    "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당하며, 금융감독원 내에 설치되어 이용자와 금융기관 간의 분쟁을 공정하고 신속하게 해결하는 역할을 수행합니다.",
                    "금융감독원 내 전자금융분쟁조정위원회가 전자금융거래에서 발생하는 이용자 피해에 대한 분쟁조정 신청을 접수하고 처리하는 업무를 담당하고 있습니다."
                ],
                "방안_묻기": [
                    "전자금융거래법에 따라 접근매체 보안을 강화하고 전자서명 및 인증체계를 고도화하며, 거래내역 실시간 통지와 이상거래 탐지시스템 구축을 통한 종합적인 이용자 보호방안이 필요합니다."
                ]
            },
            "일반": {
                "일반": [
                    "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 통해 안전하고 효율적인 운영을 수행하며, 정기적인 평가와 개선을 통해 서비스 품질을 향상시켜야 합니다.",
                    "전문적인 기술과 절차를 바탕으로 예방 조치를 강화하고 실시간 관리 체계를 구축하여 위험을 최소화하며, 지속적인 교육과 훈련을 통해 전문성을 향상시켜야 합니다.",
                    "법적 요구사항을 충실히 이행하고 효과적인 관리 조치를 시행하며, 관련 기관과의 협력을 통해 종합적이고 체계적인 관리 체계를 운영해야 합니다."
                ]
            }
        }

        self.domain_keywords = {
            "사이버보안": ["트로이", "악성코드", "보안", "탐지", "대응"],
            "개인정보보호": ["개인정보", "정보주체", "보호", "동의", "처리"],
            "전자금융": ["전자금융", "분쟁조정", "접근매체", "인증", "보안"],
            "일반": ["법령", "규정", "관리", "조치", "절차"]
        }

        self.korean_financial_terms = {}
        self.institution_database = {}
        self.mc_answer_patterns = {}

    def get_smart_template_examples(
        self, domain: str, intent_type: str = "일반", question_context: str = ""
    ) -> List[str]:
        """스마트 템플릿 예시 반환 - 대폭 강화"""

        self.template_usage_stats["total_requests"] += 1
        
        # 도메인과 의도 통계 업데이트
        if domain not in self.template_usage_stats["domain_distribution"]:
            self.template_usage_stats["domain_distribution"][domain] = 0
        self.template_usage_stats["domain_distribution"][domain] += 1
        
        if intent_type not in self.template_usage_stats["intent_distribution"]:
            self.template_usage_stats["intent_distribution"][intent_type] = 0
        self.template_usage_stats["intent_distribution"][intent_type] += 1

        # 1단계: 정확한 매칭 템플릿 수집
        primary_templates = self._get_exact_match_templates(domain, intent_type)
        
        # 2단계: 컨텍스트 기반 템플릿 필터링
        if question_context:
            context_filtered_templates = self._filter_templates_by_context(
                primary_templates, question_context
            )
            if context_filtered_templates:
                primary_templates = context_filtered_templates

        # 3단계: 유사 도메인 템플릿 수집
        secondary_templates = self._get_similar_domain_templates(domain, intent_type)
        
        # 4단계: 크로스 의도 템플릿 수집
        cross_intent_templates = self._get_cross_intent_templates(domain, intent_type)
        
        # 5단계: 스마트 템플릿 선택 및 융합
        selected_templates = self._smart_template_selection(
            primary_templates, secondary_templates, cross_intent_templates, 
            domain, intent_type, question_context
        )
        
        if selected_templates:
            self.template_usage_stats["template_provided"] += 1
            self.template_usage_stats["smart_selection_used"] += 1
            
            # 품질 점수 기록
            avg_quality = sum(
                self._calculate_template_quality_score(t) for t in selected_templates
            ) / len(selected_templates)
            self.template_usage_stats["quality_scores"].append(avg_quality)
        
        return selected_templates

    def _get_exact_match_templates(self, domain: str, intent_type: str) -> List[str]:
        """정확한 매칭 템플릿 수집"""
        templates = []
        
        if domain in self.korean_subjective_templates:
            domain_templates = self.korean_subjective_templates[domain]

            if isinstance(domain_templates, dict):
                if intent_type in domain_templates:
                    templates = domain_templates[intent_type]
                elif "일반" in domain_templates:
                    templates = domain_templates["일반"]
            else:
                templates = domain_templates

        return templates if isinstance(templates, list) else []

    def _filter_templates_by_context(self, templates: List[str], context: str) -> List[str]:
        """컨텍스트 기반 템플릿 필터링"""
        if not templates or not context:
            return templates
        
        context_lower = context.lower()
        filtered_templates = []
        
        # 컨텍스트 키워드 기반 필터링
        context_keywords = {
            "트로이": ["트로이", "악성코드", "원격제어", "위장"],
            "딥페이크": ["딥페이크", "가짜", "인공지능", "머신러닝"],
            "전자금융": ["전자금융", "분쟁", "조정", "접근매체"],
            "개인정보": ["개인정보", "정보주체", "보호", "침해"],
            "한국은행": ["한국은행", "자료제출", "통화신용"],
        }
        
        for keyword, related_terms in context_keywords.items():
            if keyword in context_lower:
                for template in templates:
                    if any(term in template for term in related_terms):
                        filtered_templates.append(template)
                break
        
        return filtered_templates if filtered_templates else templates

    def _get_similar_domain_templates(self, domain: str, intent_type: str) -> List[str]:
        """유사 도메인 템플릿 수집"""
        similar_domains = {
            "사이버보안": ["정보보안", "위험관리"],
            "정보보안": ["사이버보안", "위험관리"],
            "개인정보보호": ["정보보안", "사이버보안"],
            "전자금융": ["금융투자", "위험관리"],
            "금융투자": ["전자금융", "위험관리"],
            "위험관리": ["정보보안", "사이버보안", "금융투자"],
        }
        
        templates = []
        for similar_domain in similar_domains.get(domain, []):
            if similar_domain in self.korean_subjective_templates:
                similar_templates = self.korean_subjective_templates[similar_domain]
                if isinstance(similar_templates, dict) and intent_type in similar_templates:
                    templates.extend(similar_templates[intent_type][:2])  # 최대 2개
        
        return templates

    def _get_cross_intent_templates(self, domain: str, intent_type: str) -> List[str]:
        """크로스 의도 템플릿 수집"""
        if domain not in self.korean_subjective_templates:
            return []
        
        domain_templates = self.korean_subjective_templates[domain]
        if not isinstance(domain_templates, dict):
            return []
        
        # 의도 우선순위
        intent_priority = ["특징_묻기", "방안_묻기", "지표_묻기", "기관_묻기", "절차_묻기", "조치_묻기"]
        
        templates = []
        for priority_intent in intent_priority:
            if priority_intent != intent_type and priority_intent in domain_templates:
                if domain_templates[priority_intent]:
                    templates.extend(domain_templates[priority_intent][:1])  # 각각 1개씩
                    if len(templates) >= 3:  # 최대 3개
                        break
        
        return templates

    def _smart_template_selection(
        self, 
        primary: List[str], 
        secondary: List[str], 
        cross_intent: List[str],
        domain: str,
        intent_type: str,
        context: str = ""
    ) -> List[str]:
        """스마트 템플릿 선택 및 융합"""
        
        all_templates = []
        
        # 1차: 고품질 primary 템플릿
        high_quality_primary = [
            t for t in primary 
            if self._calculate_template_quality_score(t) >= 0.7
        ]
        all_templates.extend(high_quality_primary[:3])  # 최대 3개
        
        # 2차: secondary 템플릿 보완
        if len(all_templates) < 3:
            quality_secondary = [
                t for t in secondary 
                if self._calculate_template_quality_score(t) >= 0.6
            ]
            remaining_slots = 3 - len(all_templates)
            all_templates.extend(quality_secondary[:remaining_slots])
        
        # 3차: cross_intent 템플릿 추가
        if len(all_templates) < 3:
            quality_cross = [
                t for t in cross_intent 
                if self._calculate_template_quality_score(t) >= 0.6
            ]
            remaining_slots = 3 - len(all_templates)
            all_templates.extend(quality_cross[:remaining_slots])
        
        # 4차: 부족한 경우 동적 생성
        if len(all_templates) < 2:
            dynamic_templates = self._generate_dynamic_templates(domain, intent_type, context)
            all_templates.extend(dynamic_templates[:2])
        
        # 5차: 최종 품질 검증 및 정렬
        final_templates = []
        for template in all_templates:
            if (template and 
                len(template) >= 20 and 
                not self._has_repetitive_pattern(template)):
                final_templates.append(template)
        
        # 품질 점수순 정렬
        final_templates.sort(
            key=lambda t: self._calculate_template_quality_score(t), 
            reverse=True
        )
        
        return final_templates[:5]  # 최대 5개 반환

    def _generate_dynamic_templates(
        self, domain: str, intent_type: str, context: str = ""
    ) -> List[str]:
        """동적 템플릿 생성 - 향상된 버전"""
        
        # 도메인별 기본 프레임워크
        domain_frameworks = {
            "사이버보안": {
                "주체": "사이버보안 위협",
                "방법": "다층 방어체계와 실시간 모니터링",
                "목적": "시스템 보호와 침해 예방",
                "법령": "정보통신망법"
            },
            "개인정보보호": {
                "주체": "개인정보",
                "방법": "수집 최소화와 보호조치",
                "목적": "정보주체 권리 보장",
                "법령": "개인정보보호법"
            },
            "전자금융": {
                "주체": "전자금융거래",
                "방법": "접근매체 보안과 인증체계",
                "목적": "이용자 보호와 안전한 거래",
                "법령": "전자금융거래법"
            },
            "일반": {
                "주체": "해당 분야",
                "방법": "체계적인 관리와 지속적인 개선",
                "목적": "안전하고 효율적인 운영",
                "법령": "관련 법령"
            }
        }
        
        framework = domain_frameworks.get(domain, domain_frameworks["일반"])
        
        # 의도별 동적 템플릿 생성
        dynamic_templates = []
        
        if intent_type == "특징_묻기":
            dynamic_templates = [
                f"{framework['주체']}의 주요 특징은 {framework['방법']}을 통해 {framework['목적']}을 달성하는 특성을 가집니다.",
                f"핵심적인 특징으로는 전문적 기술과 체계적 절차를 바탕으로 {framework['목적']}을 수행하는 특성이 있습니다."
            ]
        elif intent_type == "지표_묻기":
            dynamic_templates = [
                f"주요 탐지 지표로는 {framework['방법']}을 활용한 모니터링을 통해 이상 징후를 조기에 식별할 수 있습니다.",
                f"실시간 모니터링과 정기적 분석을 통해 {framework['주체']} 관련 위험 신호를 탐지할 수 있습니다."
            ]
        elif intent_type == "방안_묻기":
            dynamic_templates = [
                f"{framework['주체']}에 대한 효과적인 대응방안으로는 {framework['방법']}을 통한 {framework['목적']}이 필요합니다.",
                f"{framework['법령']}에 따라 예방, 탐지, 대응, 복구의 단계별 방안을 수립하여 종합적으로 관리해야 합니다."
            ]
        elif intent_type == "기관_묻기":
            dynamic_templates = [
                f"관련 전문기관에서 {framework['법령']}에 따라 {framework['주체']} 관련 업무를 담당하고 있습니다.",
                f"해당 분야의 소관 기관에서 {framework['목적']}을 위한 관리 및 감독 업무를 수행합니다."
            ]
        else:
            dynamic_templates = [
                f"{framework['주체']} 분야에서는 {framework['법령']}에 따라 {framework['방법']}을 통한 {framework['목적']}이 중요합니다.",
                f"체계적인 관리와 지속적인 개선을 통해 {framework['목적']}을 달성해야 합니다."
            ]
        
        return dynamic_templates

    def get_template_examples(
        self, domain: str, intent_type: str = "일반"
    ) -> List[str]:
        """템플릿 예시 반환 (기존 호환성 유지)"""
        return self.get_smart_template_examples(domain, intent_type)

    def get_enhanced_template_structure(
        self, domain: str, intent_type: str, question: str = ""
    ) -> Dict:
        """향상된 템플릿 구조 반환"""
        
        # 기본 구조 패턴 가져오기
        if intent_type in self.answer_structure_patterns:
            base_pattern = self.answer_structure_patterns[intent_type]
        else:
            base_pattern = {
                "opening": ["관련"],
                "content": ["을 통해"],
                "closing": ["필요합니다"],
                "connectors": ["또한"],
                "professional_terms": ["관리", "조치"]
            }
        
        # 질문 컨텍스트 기반 구조 조정
        enhanced_structure = base_pattern.copy()
        
        if question:
            # 질문에서 키워드 추출하여 구조 조정
            if "트로이" in question or "악성코드" in question:
                enhanced_structure["domain_specific"] = ["악성코드", "원격제어", "위장", "침투"]
            elif "딥페이크" in question:
                enhanced_structure["domain_specific"] = ["딥페이크", "인공지능", "탐지", "인증"]
            elif "전자금융" in question:
                enhanced_structure["domain_specific"] = ["전자금융", "분쟁조정", "접근매체", "보안"]
            elif "개인정보" in question:
                enhanced_structure["domain_specific"] = ["개인정보", "정보주체", "보호조치", "동의"]
        
        # 자연스러운 문장 연결 패턴 추가
        enhanced_structure["natural_flows"] = [
            "이는", "그 결과", "따라서", "또한", "더불어", "아울러"
        ]
        
        return enhanced_structure

    def create_template_fusion_guide(
        self, templates: List[str], intent_type: str
    ) -> Dict:
        """템플릿 융합 가이드 생성"""
        
        if not templates:
            return {}
        
        fusion_guide = {
            "structure_hints": [],
            "key_phrases": [],
            "connection_patterns": [],
            "professional_vocabulary": [],
            "length_guidance": "",
            "tone_guidance": ""
        }
        
        # 구조 힌트 추출
        for template in templates:
            # 시작 패턴 분석
            if template.startswith(("주요", "핵심", "해당")):
                fusion_guide["structure_hints"].append("명확한 주제 제시로 시작")
            
            # 길이 분석
            length = len(template)
            if 50 <= length <= 150:
                fusion_guide["length_guidance"] = "간결하고 명확한 1-2문장 구성"
            elif 150 < length <= 300:
                fusion_guide["length_guidance"] = "상세한 설명을 포함한 2-3문장 구성"
        
        # 핵심 구문 추출
        common_phrases = self._extract_common_phrases(templates)
        fusion_guide["key_phrases"] = common_phrases[:5]
        
        # 연결 패턴 분석
        connection_words = ["또한", "더불어", "그리고", "아울러", "특히"]
        for template in templates:
            for word in connection_words:
                if word in template:
                    fusion_guide["connection_patterns"].append(word)
        
        # 전문 용어 추출
        professional_terms = [
            "법령", "규정", "조치", "관리", "보안", "방안", "절차", "기준",
            "정책", "체계", "시스템", "위원회", "기관", "업무", "담당"
        ]
        for template in templates:
            for term in professional_terms:
                if term in template and term not in fusion_guide["professional_vocabulary"]:
                    fusion_guide["professional_vocabulary"].append(term)
        
        # 톤 가이드
        if intent_type == "기관_묻기":
            fusion_guide["tone_guidance"] = "공식적이고 정확한 톤"
        elif intent_type == "특징_묻기":
            fusion_guide["tone_guidance"] = "설명적이고 논리적인 톤"
        elif intent_type == "방안_묻기":
            fusion_guide["tone_guidance"] = "실무적이고 구체적인 톤"
        else:
            fusion_guide["tone_guidance"] = "전문적이고 중립적인 톤"
        
        return fusion_guide

    def _extract_common_phrases(self, templates: List[str]) -> List[str]:
        """공통 구문 추출"""
        if not templates:
            return []
        
        # 3-7자 구문 추출
        phrase_counts = {}
        for template in templates:
            words = template.split()
            for i in range(len(words) - 1):
                for j in range(i + 2, min(i + 4, len(words) + 1)):
                    phrase = " ".join(words[i:j])
                    if 3 <= len(phrase) <= 15:
                        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # 2회 이상 등장하는 구문만 반환
        common_phrases = [phrase for phrase, count in phrase_counts.items() if count >= 2]
        return sorted(common_phrases, key=lambda x: phrase_counts[x], reverse=True)

    def analyze_question(self, question: str) -> Dict:
        """질문 분석 - 템플릿 선택 강화"""
        question_lower = question.lower()

        # 기본 분석
        analysis_result = {
            "domain": self._detect_primary_domain(question_lower),
            "secondary_domains": self._detect_secondary_domains(question_lower),
            "complexity": self._calculate_complexity(question),
            "technical_level": "중급",
            "korean_technical_terms": self._find_korean_technical_terms(question),
            "compliance": self._check_competition_compliance(question),
            "institution_info": self._check_institution_question_enhanced(question),
            "mc_pattern_info": self._analyze_mc_pattern(question),
            "template_selection_hints": self._get_template_selection_hints(question),
        }

        # 기술 수준 결정
        analysis_result["technical_level"] = self._determine_technical_level(
            analysis_result["complexity"], 
            analysis_result["korean_technical_terms"]
        )

        return analysis_result

    def _detect_primary_domain(self, question_lower: str) -> List[str]:
        """주요 도메인 감지"""
        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    # 핵심 키워드 가중치 적용
                    if keyword in ["트로이", "RAT", "원격제어", "전자금융분쟁조정위원회", "개인정보보호위원회"]:
                        score += 3
                    else:
                        score += 1

            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            # 상위 도메인들 반환
            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
            return [domain for domain, score in sorted_domains if score >= sorted_domains[0][1] * 0.5]
        else:
            return ["일반"]

    def _detect_secondary_domains(self, question_lower: str) -> List[str]:
        """보조 도메인 감지"""
        # 연관 도메인 매핑
        domain_relations = {
            "사이버보안": ["정보보안", "위험관리"],
            "개인정보보호": ["정보보안", "사이버보안"],
            "전자금융": ["금융투자", "위험관리"],
            "금융투자": ["전자금융", "위험관리"],
            "정보보안": ["사이버보안", "위험관리"],
            "위험관리": ["정보보안", "사이버보안"]
        }
        
        primary_domains = self._detect_primary_domain(question_lower)
        secondary_domains = []
        
        for primary in primary_domains:
            if primary in domain_relations:
                secondary_domains.extend(domain_relations[primary])
        
        return list(set(secondary_domains))

    def _get_template_selection_hints(self, question: str) -> Dict:
        """템플릿 선택 힌트"""
        hints = {
            "context_keywords": [],
            "preferred_style": "formal",
            "length_preference": "medium",
            "specificity_level": "detailed",
        }
        
        question_lower = question.lower()
        
        # 컨텍스트 키워드 추출
        context_mapping = {
            "트로이": ["악성코드", "원격제어", "위장", "침투"],
            "딥페이크": ["가짜", "인공지능", "탐지", "인증"],
            "전자금융": ["분쟁", "조정", "접근매체", "보안"],
            "개인정보": ["정보주체", "보호", "침해", "동의"],
            "한국은행": ["자료제출", "통화신용", "금융기관"]
        }
        
        for key, contexts in context_mapping.items():
            if key in question_lower:
                hints["context_keywords"] = contexts
                break
        
        # 스타일 선호도
        if any(word in question_lower for word in ["기관", "위원회", "담당"]):
            hints["preferred_style"] = "official"
        elif any(word in question_lower for word in ["특징", "원리", "방식"]):
            hints["preferred_style"] = "explanatory"
        elif any(word in question_lower for word in ["방안", "대책", "조치"]):
            hints["preferred_style"] = "practical"
        
        # 길이 선호도
        if len(question) > 200:
            hints["length_preference"] = "detailed"
        elif len(question) < 100:
            hints["length_preference"] = "concise"
        
        return hints

    def _check_institution_question_enhanced(self, question: str) -> Dict:
        """향상된 기관 질문 확인"""
        question_lower = question.lower()

        institution_info = {
            "is_institution_question": False,
            "institution_type": None,
            "relevant_institution": None,
            "confidence": 0.0,
            "question_pattern": None,
            "hint_available": False,
            "specific_context": None,
        }

        # 확장된 기관 질문 패턴
        institution_patterns = [
            "기관.*기술하세요", "기관.*설명하세요", "어떤.*기관", "어느.*기관",
            "조정.*신청.*기관", "분쟁.*조정.*기관", "신청.*수.*있는.*기관",
            "담당.*기관", "관리.*기관", "감독.*기관", "소관.*기관",
            "신고.*기관", "접수.*기관", "상담.*기관", "문의.*기관",
            "위원회.*무엇", "위원회.*어디", "위원회.*설명", "어떤.*위원회",
            "분쟁.*어디", "신고.*어디", "상담.*어디", "업무.*담당",
            "책임.*기관", "주관.*기관", "관할.*기관", "설치.*기관"
        ]

        pattern_matches = 0
        matched_pattern = None
        for pattern in institution_patterns:
            if re.search(pattern, question_lower):
                pattern_matches += 1
                matched_pattern = pattern

        is_asking_institution = pattern_matches > 0

        if is_asking_institution:
            institution_info["is_institution_question"] = True
            institution_info["confidence"] = min(pattern_matches / 1.0, 1.0)
            institution_info["question_pattern"] = matched_pattern
            institution_info["hint_available"] = True

            # 구체적 컨텍스트 분석
            if "전자금융" in question_lower and "분쟁" in question_lower:
                institution_info["institution_type"] = "전자금융분쟁조정"
                institution_info["specific_context"] = "전자금융거래 분쟁조정"
            elif "개인정보" in question_lower and ("침해" in question_lower or "신고" in question_lower):
                institution_info["institution_type"] = "개인정보보호"
                institution_info["specific_context"] = "개인정보 침해신고 및 상담"
            elif "한국은행" in question_lower or "자료제출" in question_lower:
                institution_info["institution_type"] = "한국은행"
                institution_info["specific_context"] = "통화신용정책 관련 자료제출"

        return institution_info

    # 기존 메서드들 유지 (호환성을 위해)
    def get_template_hints(self, domain: str, intent_type: str = "일반") -> str:
        """템플릿 힌트 반환 (기존 호환성)"""
        structure = self.get_enhanced_template_structure(domain, intent_type)
        
        hints = []
        if "opening" in structure:
            hints.append(f"시작: {', '.join(structure['opening'][:2])}")
        if "professional_terms" in structure:
            hints.append(f"전문용어: {', '.join(structure['professional_terms'][:3])}")
        
        return " / ".join(hints)

    def get_institution_hints(self, institution_type: str) -> str:
        """기관별 힌트 정보 반환 (기존 호환성)"""
        if institution_type in self.institution_database:
            info = self.institution_database[institution_type]
            hint_parts = []

            if "기관명" in info:
                hint_parts.append(f"기관명: {info['기관명']}")
            if "역할" in info:
                hint_parts.append(f"주요 역할: {info['역할']}")

            return " ".join(hint_parts)

        # 기본 힌트
        default_hints = {
            "전자금융분쟁조정": "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다.",
            "개인정보보호": "개인정보보호위원회에서 개인정보 보호에 관한 업무를 총괄합니다.",
            "한국은행": "한국은행에서 통화신용정책 수행과 지급결제제도 운영을 담당합니다.",
        }

        return default_hints.get(institution_type, "해당 분야의 전문 기관에서 관련 업무를 담당하고 있습니다.")

    # 기존 메서드들 (변경 없음)
    def get_mc_pattern_hints(self, question: str) -> str:
        """객관식 패턴 힌트 반환"""
        mc_pattern_info = self._analyze_mc_pattern(question)
        if mc_pattern_info["is_mc_question"] and mc_pattern_info["pattern_confidence"] > 0.5:
            pattern_key = mc_pattern_info["pattern_key"]
            if pattern_key in self.mc_answer_patterns:
                pattern_data = self.mc_answer_patterns[pattern_key]
                return f"이 문제는 {pattern_data.get('explanation', '관련 내용')}에 대한 문제입니다."
        return None

    def _analyze_mc_pattern(self, question: str) -> Dict:
        """객관식 패턴 분석"""
        return {
            "is_mc_question": False,
            "pattern_type": None,
            "pattern_confidence": 0.0,
            "pattern_key": None,
            "hint_available": False,
        }

    def _calculate_complexity(self, question: str) -> float:
        """질문 복잡도 계산"""
        length_factor = min(len(question) / 200, 1.0)
        korean_term_count = sum(1 for term in self.korean_financial_terms.keys() if term in question)
        term_factor = min(korean_term_count / 3, 1.0)
        return (length_factor + term_factor) / 2

    def _find_korean_technical_terms(self, question: str) -> List[str]:
        """한국어 전문 용어 찾기"""
        found_terms = []
        for term in self.korean_financial_terms.keys():
            if term in question:
                found_terms.append(term)
        return found_terms

    def _determine_technical_level(self, complexity: float, korean_terms: List[str]) -> str:
        """기술 수준 결정"""
        if complexity > 0.7 or len(korean_terms) >= 2:
            return "고급"
        elif complexity > 0.4 or len(korean_terms) >= 1:
            return "중급"
        else:
            return "초급"

    def _check_competition_compliance(self, question: str) -> Dict:
        """대회 규칙 준수 확인"""
        return {
            "korean_content": True,
            "appropriate_domain": True,
            "no_external_dependency": True,
        }

    def get_analysis_statistics(self) -> Dict:
        """분석 통계 반환"""
        stats = {
            "korean_terms_available": len(self.korean_financial_terms),
            "institutions_available": len(self.institution_database),
            "template_domains": len(self.korean_subjective_templates),
            "template_usage_stats": self.template_usage_stats.copy(),
        }
        
        # 품질 점수 통계 추가
        if self.template_usage_stats["quality_scores"]:
            scores = self.template_usage_stats["quality_scores"]
            stats["quality_statistics"] = {
                "average_quality": sum(scores) / len(scores),
                "min_quality": min(scores),
                "max_quality": max(scores),
                "total_evaluations": len(scores)
            }
        
        return stats

    def cleanup(self):
        """정리"""
        pass