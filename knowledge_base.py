# knowledge_base.py

"""
금융보안 지식베이스
- 도메인별 키워드 분류
- 전문 용어 처리
- 한국어 전용 답변 템플릿 제공
"""

import pickle
import os
from datetime import datetime
from typing import Dict, List
import random

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
                "원격제어", "원격접근", "봇넷", "분산서비스거부공격", 
                "지능형지속위협", "제로데이", "딥페이크", "사회공학", "취약점", "패치"
            ],
            "정보보안": [
                "정보보안", "보안관리", "정보보안관리체계", "보안정책", 
                "접근통제", "암호화", "방화벽", "침입탐지",
                "침입방지시스템", "보안정보이벤트관리", "보안관제", "인증",
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
        
        # 한국어 전용 주관식 답변 템플릿
        self.korean_subjective_templates = {
            "사이버보안": [
                "해당 악성코드는 원격제어 기능을 통해 시스템에 침입하며 백신 프로그램과 행위 기반 탐지 시스템을 활용하여 탐지할 수 있습니다. 주요 대응방안으로는 네트워크 모니터링 강화와 접근권한 관리를 통한 예방조치가 있습니다.",
                "사이버보안 위협에 대응하기 위해서는 다층 방어체계를 구축하고 실시간 모니터링과 침입탐지시스템을 운영해야 합니다. 또한 정기적인 보안교육과 훈련을 실시하여 보안 인식을 제고해야 합니다.",
                "보안정책을 수립하고 정기적인 보안교육과 훈련을 실시하며 취약점 점검과 보안패치를 지속적으로 수행해야 합니다. 특히 사용자 계정 관리와 접근권한 통제를 강화하여 내부 보안을 확보해야 합니다."
            ],
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
        
        # 한국어 전용 금융 전문 용어 사전
        self.korean_financial_terms = {
            "정보보안관리체계": "조직의 정보자산을 보호하기 위한 종합적인 관리체계",
            "개인정보관리체계": "개인정보의 안전한 처리를 위한 체계적 관리방안",
            "원격접근": "네트워크를 통해 원격지에서 컴퓨터 시스템에 접근하는 방식",
            "지능형지속위협": "특정 목표를 대상으로 장기간에 걸쳐 수행되는 고도화된 사이버공격",
            "데이터유출방지": "조직 내부의 중요 데이터가 외부로 유출되는 것을 방지하는 보안기술",
            "모바일기기관리": "조직에서 사용하는 모바일 기기의 보안을 관리하는 솔루션",
            "보안정보이벤트관리": "보안 정보와 이벤트를 통합적으로 관리하고 분석하는 시스템"
        }
        
        # 질문 분석 이력
        self.analysis_history = {
            "domain_frequency": {},
            "complexity_distribution": {},
            "question_patterns": []
        }
        
        # 이전 분석 이력 로드
        self._load_analysis_history()
    
    def _load_analysis_history(self):
        """이전 분석 이력 로드"""
        history_file = "./analysis_history.pkl"
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'rb') as f:
                    saved_history = pickle.load(f)
                    self.analysis_history.update(saved_history)
            except Exception:
                pass
    
    def _save_analysis_history(self):
        """분석 이력 저장"""
        history_file = "./analysis_history.pkl"
        
        try:
            save_data = {
                **self.analysis_history,
                "last_updated": datetime.now().isoformat()
            }
            
            # 최근 1000개 패턴만 저장
            save_data["question_patterns"] = save_data["question_patterns"][-1000:]
            
            with open(history_file, 'wb') as f:
                pickle.dump(save_data, f)
        except Exception:
            pass
    
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
        
        # 한국어 전문 용어 포함 여부
        korean_terms = self._find_korean_technical_terms(question)
        
        # 분석 결과 저장
        analysis_result = {
            "domain": detected_domains,
            "complexity": complexity,
            "technical_level": self._determine_technical_level(complexity, korean_terms),
            "korean_technical_terms": korean_terms
        }
        
        # 이력에 추가
        self._add_to_analysis_history(question, analysis_result)
        
        return analysis_result
    
    def _add_to_analysis_history(self, question: str, analysis: Dict):
        """분석 이력에 추가"""
        # 도메인 빈도 업데이트
        for domain in analysis["domain"]:
            self.analysis_history["domain_frequency"][domain] = \
                self.analysis_history["domain_frequency"].get(domain, 0) + 1
        
        # 복잡도 분포 업데이트
        level = analysis["technical_level"]
        self.analysis_history["complexity_distribution"][level] = \
            self.analysis_history["complexity_distribution"].get(level, 0) + 1
        
        # 질문 패턴 추가
        pattern = {
            "question_length": len(question),
            "domain": analysis["domain"][0] if analysis["domain"] else "일반",
            "complexity": analysis["complexity"],
            "korean_terms_count": len(analysis["korean_technical_terms"]),
            "timestamp": datetime.now().isoformat()
        }
        
        self.analysis_history["question_patterns"].append(pattern)
    
    def get_korean_subjective_template(self, domain: str) -> str:
        """한국어 주관식 답변 템플릿 반환"""
        
        if domain in self.korean_subjective_templates:
            templates = self.korean_subjective_templates[domain]
        else:
            templates = self.korean_subjective_templates["일반"]
        
        return random.choice(templates)
    
    def get_subjective_template(self, domain: str) -> str:
        """주관식 답변 템플릿 반환 (한국어 전용)"""
        return self.get_korean_subjective_template(domain)
    
    def _calculate_complexity(self, question: str) -> float:
        """질문 복잡도 계산"""
        # 길이 기반 복잡도
        length_factor = min(len(question) / 200, 1.0)
        
        # 한국어 전문 용어 개수
        korean_term_count = sum(1 for term in self.korean_financial_terms.keys() 
                               if term in question)
        term_factor = min(korean_term_count / 3, 1.0)
        
        # 도메인 개수
        domain_count = sum(1 for keywords in self.domain_keywords.values() 
                          if any(keyword in question.lower() for keyword in keywords))
        domain_factor = min(domain_count / 2, 1.0)
        
        return (length_factor + term_factor + domain_factor) / 3
    
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
    
    def get_domain_specific_guidance(self, domain: str) -> Dict:
        """도메인별 지침 반환"""
        guidance = {
            "개인정보보호": {
                "key_laws": ["개인정보보호법", "정보통신망법"],
                "key_concepts": ["정보주체", "개인정보처리자", "동의", "목적외이용금지"],
                "oversight_body": "개인정보보호위원회"
            },
            "전자금융": {
                "key_laws": ["전자금융거래법", "전자서명법"],
                "key_concepts": ["접근매체", "전자서명", "인증", "분쟁조정"],
                "oversight_body": "금융감독원, 한국은행"
            },
            "사이버보안": {
                "key_laws": ["정보통신망법", "개인정보보호법"],
                "key_concepts": ["악성코드", "침입탐지", "보안관제", "사고대응"],
                "oversight_body": "과학기술정보통신부, 경찰청"
            },
            "정보보안": {
                "key_laws": ["정보통신망법", "전자정부법"],
                "key_concepts": ["정보보안관리체계", "접근통제", "암호화", "백업"],
                "oversight_body": "과학기술정보통신부"
            },
            "금융투자": {
                "key_laws": ["자본시장법", "금융투자업규정"],
                "key_concepts": ["투자자보호", "적합성원칙", "설명의무", "내부통제"],
                "oversight_body": "금융감독원, 금융위원회"
            },
            "위험관리": {
                "key_laws": ["은행법", "보험업법", "자본시장법"],
                "key_concepts": ["위험평가", "내부통제", "컴플라이언스", "감사"],
                "oversight_body": "금융감독원"
            }
        }
        
        return guidance.get(domain, {
            "key_laws": ["관련 법령"],
            "key_concepts": ["체계적 관리", "지속적 개선"],
            "oversight_body": "관계기관"
        })
    
    def get_analysis_statistics(self) -> Dict:
        """분석 통계 반환"""
        return {
            "domain_frequency": dict(self.analysis_history["domain_frequency"]),
            "complexity_distribution": dict(self.analysis_history["complexity_distribution"]),
            "total_analyzed": len(self.analysis_history["question_patterns"]),
            "korean_terms_available": len(self.korean_financial_terms)
        }
    
    def cleanup(self):
        """정리"""
        self._save_analysis_history()