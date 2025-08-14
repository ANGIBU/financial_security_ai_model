# knowledge_base.py

"""
금융보안 지식베이스
- 도메인별 키워드 분류
- 전문 용어 처리
- 답변 템플릿 제공
"""

from typing import Dict, List

class FinancialSecurityKnowledgeBase:
    """금융보안 지식베이스"""
    
    def __init__(self):
        # 도메인별 키워드
        self.domain_keywords = {
            "개인정보보호": [
                "개인정보", "정보주체", "개인정보보호법", "민감정보", 
                "고유식별정보", "수집", "이용", "제공", "파기", "동의",
                "법정대리인", "아동", "처리", "개인정보처리방침", "열람권",
                "정정삭제권", "처리정지권", "손해배상", "개인정보보호위원회"
            ],
            "전자금융": [
                "전자금융", "전자적", "접근매체", "전자금융거래법", 
                "전자서명", "전자인증", "공인인증서", "전자금융업",
                "전자지급수단", "전자화폐", "전자금융거래", "인증",
                "전자금융분쟁조정위원회", "금융감독원", "한국은행"
            ],
            "사이버보안": [
                "트로이", "악성코드", "해킹", "멀웨어", "피싱", 
                "스미싱", "랜섬웨어", "바이러스", "웜", "스파이웨어",
                "원격제어", "RAT", "봇넷", "DDoS", "APT", "제로데이",
                "딥페이크", "소셜엔지니어링", "취약점", "패치"
            ],
            "정보보안": [
                "정보보안", "보안관리", "ISMS", "보안정책", 
                "접근통제", "암호화", "방화벽", "침입탐지",
                "IDS", "IPS", "SIEM", "보안관제", "인증",
                "권한관리", "로그관리", "백업", "복구", "재해복구"
            ],
            "금융투자": [
                "금융투자업", "투자자문업", "투자매매업", "투자중개업",
                "집합투자업", "신탁업", "소비자금융업", "보험중개업",
                "금융투자회사", "자본시장법", "펀드", "파생상품"
            ],
            "위험관리": [
                "위험관리", "위험평가", "위험대응", "위험수용", "위험회피",
                "위험전가", "위험감소", "위험분석", "위험식별", "위험모니터링",
                "리스크", "내부통제", "컴플라이언스", "감사"
            ]
        }
        
        # 객관식 질문 패턴
        self.mc_patterns = [
            "해당하지.*않는.*것",
            "적절하지.*않는.*것", 
            "옳지.*않는.*것",
            "틀린.*것",
            "맞는.*것",
            "옳은.*것",
            "적절한.*것",
            "올바른.*것",
            "가장.*적절한.*것",
            "가장.*옳은.*것"
        ]
        
        # 주관식 답변 템플릿
        self.subjective_templates = {
            "사이버보안": [
                "해당 악성코드는 {특징}을 가지며, {탐지방법}을 통해 탐지할 수 있습니다. 주요 대응방안으로는 {대응방안}이 있습니다.",
                "주요 특징으로는 {특징1}, {특징2} 등이 있으며, 탐지를 위해서는 {탐지방법}을 활용해야 합니다.",
                "{위협요소}에 대한 보안 위협으로는 {위협내용}이 있으며, 이에 대한 대응책으로 {대응책}을 수립해야 합니다."
            ],
            "개인정보보호": [
                "개인정보보호법에 따라 {법적근거}를 준수해야 하며, {절차}를 거쳐 처리해야 합니다.",
                "정보주체의 권리 보장을 위해 {권리내용}을 제공하고 {보호조치}를 시행해야 합니다.",
                "관련 법령에 따른 {의무사항}을 이행하고 {관리방안}을 수립해야 합니다."
            ],
            "전자금융": [
                "전자금융거래법에 따라 {법적기준}를 충족해야 하며, {기관명}에서 관련 업무를 담당합니다.",
                "{분쟁조정기관}에서 관련 분쟁조정 업무를 수행하며, {절차}를 통해 신청할 수 있습니다.",
                "전자금융 서비스 제공 시 {보안요구사항}을 준수하고 {인증방법}을 적용해야 합니다."
            ],
            "정보보안": [
                "정보보안 관리체계 수립을 위해 {관리요소}를 포함하고 {절차}를 준수해야 합니다.",
                "보안정책 수립 시 {정책요소}를 반영하고 {관리방안}을 마련해야 합니다.",
                "{보안통제}를 구현하고 정기적인 {점검절차}를 통해 지속적으로 관리해야 합니다."
            ],
            "일반": [
                "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다.",
                "전문적인 보안 정책을 수립하고 정기적인 점검과 평가를 실시해야 합니다.", 
                "법적 요구사항을 준수하며 효과적인 보안 조치를 시행하고 교육을 실시해야 합니다.",
                "위험 요소를 식별하고 적절한 대응 방안을 마련하여 체계적으로 관리해야 합니다.",
                "관리 절차를 확립하고 정기적인 평가를 통해 지속적인 개선을 추진해야 합니다."
            ]
        }
        
        # 금융 전문 용어 사전
        self.financial_terms = {
            "SBOM": "Software Bill of Materials, 소프트웨어 구성 요소 목록",
            "ISMS": "Information Security Management System, 정보보안 관리체계",
            "PIMS": "Personal Information Management System, 개인정보 관리체계", 
            "RAT": "Remote Access Trojan, 원격 접근 트로이목마",
            "APT": "Advanced Persistent Threat, 지능형 지속 위협",
            "DLP": "Data Loss Prevention, 데이터 유출 방지",
            "MDM": "Mobile Device Management, 모바일 기기 관리",
            "SIEM": "Security Information and Event Management, 보안 정보 이벤트 관리"
        }
    
    def analyze_question(self, question: str) -> Dict:
        """질문 분석"""
        question_lower = question.lower()
        
        # 도메인 찾기
        detected_domains = []
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_domains.append(domain)
        
        if not detected_domains:
            detected_domains = ["일반"]
        
        # 복잡도 계산
        complexity = self._calculate_complexity(question)
        
        # 전문 용어 포함 여부
        technical_terms = self._find_technical_terms(question)
        
        return {
            "domain": detected_domains,
            "complexity": complexity,
            "technical_level": self._determine_technical_level(complexity, technical_terms),
            "technical_terms": technical_terms
        }
    
    def get_subjective_template(self, domain: str) -> str:
        """주관식 답변 템플릿 반환"""
        import random
        
        if domain in self.subjective_templates:
            return random.choice(self.subjective_templates[domain])
        else:
            return random.choice(self.subjective_templates["일반"])
    
    def _calculate_complexity(self, question: str) -> float:
        """질문 복잡도 계산"""
        # 길이 기반 복잡도
        length_factor = min(len(question) / 200, 1.0)
        
        # 전문 용어 개수
        term_count = sum(1 for term in self.financial_terms.keys() if term.lower() in question.lower())
        term_factor = min(term_count / 3, 1.0)
        
        # 도메인 개수
        domain_count = sum(1 for keywords in self.domain_keywords.values() 
                          if any(keyword in question.lower() for keyword in keywords))
        domain_factor = min(domain_count / 2, 1.0)
        
        return (length_factor + term_factor + domain_factor) / 3
    
    def _find_technical_terms(self, question: str) -> List[str]:
        """전문 용어 찾기"""
        found_terms = []
        question_lower = question.lower()
        
        for term in self.financial_terms.keys():
            if term.lower() in question_lower:
                found_terms.append(term)
        
        return found_terms
    
    def _determine_technical_level(self, complexity: float, technical_terms: List[str]) -> str:
        """기술 수준 결정"""
        if complexity > 0.7 or len(technical_terms) >= 2:
            return "고급"
        elif complexity > 0.4 or len(technical_terms) >= 1:
            return "중급"
        else:
            return "초급"
    
    def cleanup(self):
        """정리"""
        pass