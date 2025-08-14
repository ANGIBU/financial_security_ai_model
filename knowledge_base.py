# knowledge_base.py

"""
간단한 지식베이스
- 복잡성 제거
- 핵심 도메인 정보만
"""

from typing import Dict, List

class FinancialSecurityKnowledgeBase:
    """금융보안 지식베이스 - 단순화"""
    
    def __init__(self):
        # 도메인별 키워드
        self.domain_keywords = {
            "개인정보보호": [
                "개인정보", "정보주체", "개인정보보호법", "민감정보", 
                "고유식별정보", "수집", "이용", "제공", "파기"
            ],
            "전자금융": [
                "전자금융", "전자적", "접근매체", "전자금융거래법", 
                "전자서명", "전자인증", "공인인증서"
            ],
            "사이버보안": [
                "트로이", "악성코드", "해킹", "멀웨어", "피싱", 
                "스미싱", "랜섬웨어", "바이러스", "웜"
            ],
            "정보보안": [
                "정보보안", "보안관리", "ISMS", "보안정책", 
                "접근통제", "암호화", "방화벽", "침입탐지"
            ]
        }
    
    def analyze_question(self, question: str) -> Dict:
        """질문 분석"""
        question_lower = question.lower()
        
        # 도메인 찾기
        detected_domain = "일반"
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_domain = domain
                break
        
        return {
            "domain": [detected_domain],
            "complexity": 0.5,
            "technical_level": "중급"
        }
    
    def cleanup(self):
        """정리"""
        pass