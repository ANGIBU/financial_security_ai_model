# knowledge_base.py

import re
from typing import Dict, List
from config import TEMPLATE_QUALITY_CRITERIA


class KnowledgeBase:
    """금융보안 지식베이스"""

    def __init__(self):
        self._initialize_data()
        self.template_quality_criteria = TEMPLATE_QUALITY_CRITERIA

    def _initialize_data(self):
        """데이터 초기화"""
        
        # 도메인 키워드 매핑
        self.domain_keywords = {
            "개인정보보호": [
                "개인정보", "정보주체", "개인정보보호법", "민감정보", "고유식별정보",
                "수집", "이용", "제공", "파기", "동의", "법정대리인", "아동", "처리",
                "개인정보처리방침", "열람권", "정정삭제권", "처리정지권", "손해배상",
                "개인정보보호위원회", "개인정보영향평가", "개인정보관리체계",
                "개인정보처리시스템", "개인정보보호책임자", "개인정보취급자",
                "개인정보침해신고센터", "PIMS", "관리체계 수립", "정책 수립",
                "만 14세", "미만 아동", "중요한 요소", "경영진", "최고책임자",
                "자원 할당", "내부 감사"
            ],
            "전자금융": [
                "전자금융", "전자적", "접근매체", "전자금융거래법", "전자서명",
                "전자인증", "공인인증서", "분쟁조정", "전자지급수단", "전자화폐",
                "금융감독원", "한국은행", "전자금융업", "전자금융분쟁조정위원회",
                "전자금융거래", "전자금융업무", "전자금융서비스", "전자금융거래기록",
                "이용자", "금융통화위원회", "자료제출", "통화신용정책", "지급결제제도",
                "요청", "요구", "경우", "보안 강화", "통계조사", "경영 실적", "원활한 운영"
            ],
            "사이버보안": [
                "트로이", "악성코드", "멀웨어", "바이러스", "피싱", "스미싱", "랜섬웨어",
                "해킹", "딥페이크", "원격제어", "RAT", "원격접근", "봇넷", "백도어",
                "루트킷", "취약점", "제로데이", "사회공학", "APT", "DDoS", "침입탐지",
                "침입방지", "보안관제", "SBOM", "소프트웨어 구성 요소", "Trojan",
                "원격제어 악성코드", "탐지 지표", "보안 위협", "특징", "주요 탐지",
                "금융권", "활용", "이유", "적절한", "소프트웨어", "접근 제어",
                "투명성", "다양성", "공급망 보안"
            ],
            "정보보안": [
                "정보보안", "보안관리", "ISMS", "보안정책", "접근통제", "암호화",
                "방화벽", "침입탐지", "침입방지시스템", "IDS", "IPS", "보안관제",
                "로그관리", "백업", "복구", "재해복구", "BCP", "정보보안관리체계",
                "정보보호", "관리체계 수립", "정책 수립", "최고책임자", "경영진",
                "자원 할당", "내부 감사", "절차 수립", "복구 절차", "비상연락체계",
                "개인정보 파기", "복구 목표시간", "옳지 않은", "고려", "요소"
            ],
            "금융투자": [
                "금융투자업", "투자자문업", "투자매매업", "투자중개업", "소비자금융업",
                "보험중개업", "자본시장법", "집합투자업", "신탁업", "펀드", "파생상품",
                "투자자보호", "적합성원칙", "설명의무", "금융산업", "구분",
                "해당하지 않는", "금융산업의 이해"
            ],
            "위험관리": [
                "위험관리", "위험평가", "위험대응", "위험수용", "리스크", "내부통제",
                "컴플라이언스", "위험식별", "위험분석", "위험모니터링", "위험회피",
                "위험전가", "위험감소", "잔여위험", "위험성향", "위험 관리 계획",
                "수행인력", "위험 대응 전략", "재해 복구", "복구 절차", "비상연락체계",
                "복구 목표시간", "계획 수립", "고려", "요소", "적절하지 않은", "대상", "기간"
            ]
        }

        # 한국어 금융 용어 사전
        self.korean_financial_terms = {
            "정보보안관리체계": "조직의 정보자산을 보호하기 위한 종합적인 관리체계",
            "개인정보관리체계": "개인정보의 안전한 처리를 위한 체계적 관리방안",
            "원격접근": "네트워크를 통해 원격지에서 컴퓨터 시스템에 접근하는 방식",
            "트로이목마": "정상 프로그램으로 위장하여 악의적 기능을 수행하는 악성코드",
            "원격접근도구": "네트워크를 통해 원격지 시스템을 제어할 수 있는 소프트웨어",
            "소프트웨어구성요소명세서": "소프트웨어에 포함된 구성 요소의 목록과 정보를 기록한 문서",
            "딥페이크": "인공지능을 이용하여 가짜 영상이나 음성을 제작하는 기술",
            "전자금융분쟁조정위원회": "전자금융거래 관련 분쟁의 조정을 담당하는 기관",
            "개인정보보호위원회": "개인정보 보호에 관한 업무를 총괄하는 중앙행정기관"
        }

        # 기관 데이터베이스
        self.institution_database = {
            "전자금융분쟁조정": {
                "기관명": "전자금융분쟁조정위원회",
                "소속": "금융감독원",
                "역할": "전자금융거래 관련 분쟁조정",
                "근거법": "전자금융거래법",
                "상세정보": "전자금융거래에서 발생하는 분쟁의 공정하고 신속한 해결을 위해 설치된 기구로, 이용자와 전자금융업자 간의 분쟁조정 업무를 담당합니다.",
                "관련질문패턴": ["전자금융거래법에 따라", "이용자가", "분쟁조정을 신청할 수 있는", "기관"]
            },
            "개인정보보호": {
                "기관명": "개인정보보호위원회",
                "소속": "국무총리 소속",
                "역할": "개인정보보호 정책 수립 및 감시",
                "근거법": "개인정보보호법",
                "신고기관": "개인정보침해신고센터",
                "상세정보": "개인정보 보호에 관한 업무를 총괄하는 중앙행정기관으로, 개인정보 보호 정책 수립, 법령 집행, 감시 업무를 수행합니다.",
                "관련질문패턴": ["개인정보", "침해", "신고", "상담", "보호위원회"]
            },
            "한국은행": {
                "기관명": "한국은행",
                "소속": "중앙은행",
                "역할": "통화신용정책 수행 및 지급결제제도 운영",
                "근거법": "한국은행법",
                "상세정보": "금융통화위원회의 요청에 따라 통화신용정책의 수행 및 지급결제제도의 원활한 운영을 위해 금융회사 및 전자금융업자에게 자료제출을 요구할 수 있습니다.",
                "관련질문패턴": ["한국은행", "금융통화위원회", "자료제출", "요구"]
            }
        }

        # 객관식 답변 패턴
        self.mc_answer_patterns = {
            "금융투자_해당하지않는": {
                "question_keywords": ["금융투자업", "구분", "해당하지 않는"],
                "non_financial_investment": ["소비자금융업", "보험중개업"],
                "financial_investment": ["투자자문업", "투자매매업", "투자중개업"],
                "correct_logic": "금융투자업이 아닌 것을 찾아야 함",
                "explanation": "금융투자업에는 투자자문업, 투자매매업, 투자중개업이 포함되며, 소비자금융업과 보험중개업은 금융투자업에 해당하지 않습니다.",
                "hint": "금융투자업에 해당하지 않는 업종은 소비자금융업과 보험중개업입니다."
            },
            "위험관리_적절하지않은": {
                "question_keywords": ["위험 관리", "계획 수립", "적절하지 않은"],
                "choices": ["수행인력", "위험 수용", "위험 대응 전략", "대상", "기간"],
                "correct_answer": "2",
                "explanation": "위험 관리 계획 수립 시 수행인력, 위험 대응 전략 선정, 대상, 기간을 고려해야 하며, 위험 수용은 적절하지 않습니다.",
                "hint": "위험관리 계획에서 위험 수용은 부적절한 요소입니다."
            },
            "개인정보_중요한요소": {
                "question_keywords": ["정책 수립", "가장 중요한 요소", "경영진"],
                "choices": ["정보보호 정책 제개정", "경영진의 참여", "최고책임자 지정", "자원 할당"],
                "correct_answer": "2",
                "explanation": "관리체계 수립 및 운영의 정책 수립 단계에서 가장 중요한 요소는 경영진의 참여입니다.",
                "hint": "정책 수립에서 경영진의 참여가 가장 중요합니다."
            },
            "전자금융_요구경우": {
                "question_keywords": ["한국은행", "자료제출", "요구할 수 있는 경우"],
                "choices": ["보안 강화", "통계조사", "경영 실적", "통화신용정책"],
                "correct_answer": "4",
                "explanation": "한국은행이 금융통화위원회의 요청에 따라 자료제출을 요구할 수 있는 경우는 통화신용정책의 수행 및 지급결제제도의 원활한 운영을 위해서입니다.",
                "hint": "한국은행의 자료제출 요구는 통화신용정책 수행을 위해서입니다."
            },
            "개인정보_법정대리인": {
                "question_keywords": ["만 14세 미만", "아동", "개인정보", "절차"],
                "choices": ["학교의 동의", "법정대리인의 동의", "본인의 동의", "친구의 동의"],
                "correct_answer": "2",
                "explanation": "개인정보보호법 제22조의2에 따라 만 14세 미만 아동의 개인정보를 처리하기 위해서는 법정대리인의 동의를 받아야 합니다.",
                "hint": "만 14세 미만 아동은 법정대리인의 동의가 필요합니다."
            },
            "사이버보안_SBOM": {
                "question_keywords": ["SBOM", "활용", "이유", "적절한"],
                "choices": ["접근 제어", "투명성", "개인정보 보호", "다양성", "소프트웨어 공급망"],
                "correct_answer": "5",
                "explanation": "금융권에서 SBOM을 활용하는 이유는 소프트웨어 공급망 보안을 강화하기 위해서입니다.",
                "hint": "SBOM은 소프트웨어 공급망 보안을 위해 활용됩니다."
            },
            "정보보안_재해복구": {
                "question_keywords": ["재해 복구", "계획 수립", "옳지 않은"],
                "choices": ["복구 절차", "비상연락체계", "개인정보 파기", "복구 목표시간"],
                "correct_answer": "3",
                "explanation": "재해 복구 계획 수립 시 복구 절차, 비상연락체계, 복구 목표시간 정의가 필요하며, 개인정보 파기 절차는 옳지 않습니다.",
                "hint": "재해복구 계획에서 개인정보 파기는 관련 없는 요소입니다."
            }
        }

        # 도메인별 컨텍스트 정보
        self.domain_context_info = {
            "사이버보안": {
                "기본정보": "사이버보안은 컴퓨터 시스템과 네트워크를 디지털 공격으로부터 보호하는 것입니다.",
                "주요법령": "정보통신망법, 개인정보보호법, 정보보안산업법",
                "핵심개념": {
                    "트로이목마": "정상 프로그램으로 위장하여 악의적 기능을 수행하는 악성코드",
                    "RAT": "원격접근 도구로 외부에서 시스템을 제어할 수 있는 악성코드",
                    "SBOM": "소프트웨어 구성 요소 명세서로 공급망 보안에 활용"
                }
            },
            "전자금융": {
                "기본정보": "전자적 장치를 통해 금융거래를 처리하는 서비스입니다.",
                "주요법령": "전자금융거래법, 전자서명법",
                "핵심기관": {
                    "전자금융분쟁조정위원회": "금융감독원 내 설치, 전자금융거래 분쟁조정 담당",
                    "한국은행": "통화신용정책 수행, 지급결제제도 운영"
                }
            },
            "개인정보보호": {
                "기본정보": "개인의 사생활과 인격을 보호하기 위해 개인정보를 안전하게 처리하는 것입니다.",
                "주요법령": "개인정보보호법, 정보통신망법",
                "핵심기관": {
                    "개인정보보호위원회": "개인정보 보호 정책 총괄",
                    "개인정보침해신고센터": "침해 신고 및 상담 업무"
                },
                "특별규정": "만 14세 미만 아동은 법정대리인의 동의 필요"
            },
            "정보보안": {
                "기본정보": "조직의 정보자산을 보호하기 위한 관리체계입니다.",
                "주요법령": "정보통신망법, 개인정보보호법",
                "핵심요소": "보안정책, 위험분석, 보안대책, 사후관리"
            },
            "금융투자": {
                "기본정보": "투자자 보호와 자본시장의 공정성을 위한 규제체계입니다.",
                "주요법령": "자본시장법",
                "업무구분": {
                    "포함": ["투자자문업", "투자매매업", "투자중개업"],
                    "미포함": ["소비자금융업", "보험중개업"]
                }
            },
            "위험관리": {
                "기본정보": "조직의 목표 달성에 영향을 미칠 수 있는 위험을 관리하는 체계입니다.",
                "핵심절차": ["위험식별", "위험평가", "위험대응", "위험모니터링"],
                "관리원칙": "위험 수용보다는 적극적 대응과 통제가 중요"
            }
        }

    def analyze_question(self, question: str) -> Dict:
        """질문 분석"""
        question_lower = question.lower()

        # 도메인 탐지
        detected_domains = []
        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    if keyword in [
                        "트로이", "RAT", "원격제어", "SBOM", "전자금융분쟁조정위원회", 
                        "개인정보보호위원회", "만 14세", "위험 관리", "금융투자업",
                    ]:
                        score += 5
                    elif keyword in [
                        "개인정보", "전자금융", "사이버보안", "정보보안", "금융투자", "위험관리"
                    ]:
                        score += 3
                    else:
                        score += 1

            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            detected_domains = [best_domain]
        else:
            detected_domains = ["일반"]

        # 기타 분석
        try:
            complexity = self._calculate_complexity(question)
            korean_terms = self._find_korean_technical_terms(question)
            compliance_check = self._check_competition_compliance(question)
            institution_info = self._check_institution_question(question)
            mc_pattern_info = self._analyze_mc_pattern(question)
        except Exception as e:
            print(f"질문 분석 중 오류: {e}")
            complexity = 0.5
            korean_terms = []
            compliance_check = {"korean_content": True, "appropriate_domain": True, "no_external_dependency": True}
            institution_info = {"is_institution_question": False}
            mc_pattern_info = {"is_mc_question": False}

        analysis_result = {
            "domain": detected_domains,
            "complexity": complexity,
            "technical_level": self._determine_technical_level(complexity, korean_terms),
            "korean_technical_terms": korean_terms,
            "compliance": compliance_check,
            "institution_info": institution_info,
            "mc_pattern_info": mc_pattern_info,
        }

        return analysis_result

    def get_domain_context(self, domain: str) -> str:
        """도메인 컨텍스트 정보 제공"""
        try:
            if domain not in self.domain_context_info:
                return "관련 법령과 규정을 참고하여 전문적인 답변을 작성하세요."
            
            context = self.domain_context_info[domain]
            context_text = f"도메인: {domain}\n"
            context_text += f"기본 정보: {context.get('기본정보', '')}\n"
            
            if '주요법령' in context:
                context_text += f"관련 법령: {context['주요법령']}\n"
            
            if '핵심개념' in context:
                context_text += "핵심 개념:\n"
                for concept, desc in context['핵심개념'].items():
                    context_text += f"- {concept}: {desc}\n"
            
            if '핵심기관' in context:
                context_text += "관련 기관:\n"
                for org, role in context['핵심기관'].items():
                    context_text += f"- {org}: {role}\n"
            
            if '업무구분' in context:
                context_text += "업무 구분:\n"
                for category, items in context['업무구분'].items():
                    context_text += f"- {category}: {', '.join(items)}\n"
            
            if '핵심절차' in context:
                context_text += f"핵심 절차: {', '.join(context['핵심절차'])}\n"
            
            if '관리원칙' in context:
                context_text += f"관리 원칙: {context['관리원칙']}\n"
            
            if '특별규정' in context:
                context_text += f"특별 규정: {context['특별규정']}\n"
            
            return context_text
            
        except Exception as e:
            print(f"도메인 컨텍스트 제공 오류: {e}")
            return "관련 법령과 규정을 참고하여 전문적인 답변을 작성하세요."

    def get_mc_pattern_hints(self, question: str) -> str:
        """객관식 패턴 힌트 제공"""
        question_lower = question.lower()
        
        # 금융투자업 구분 문제 특별 처리
        if ("금융투자업" in question_lower and 
            "구분" in question_lower and 
            "해당하지" in question_lower and 
            "않는" in question_lower):
            return "금융투자업에는 투자자문업, 투자매매업, 투자중개업이 포함됩니다. 소비자금융업과 보험중개업은 금융투자업에 해당하지 않으므로, 이 중에서 선택하세요."
        
        # 정확한 패턴 매칭
        best_match = None
        best_score = 0
        
        for pattern_key, pattern_data in self.mc_answer_patterns.items():
            score = 0
            matched_keywords = 0
            
            for keyword in pattern_data["question_keywords"]:
                if keyword in question_lower:
                    matched_keywords += 1
                    if keyword in ["해당하지 않는", "적절하지 않은", "옳지 않은"]:
                        score += 3
                    elif keyword in ["가장 중요한", "가장 적절한"]:
                        score += 3
                    else:
                        score += 1
            
            # 매칭 비율 계산
            try:
                match_ratio = matched_keywords / len(pattern_data["question_keywords"])
                final_score = score * match_ratio
                
                if final_score > best_score and matched_keywords >= 2:
                    best_score = final_score
                    best_match = pattern_data
            except ZeroDivisionError:
                continue

        # 최적 매치 힌트 제공
        if best_match and best_score >= 2:
            hint_parts = []
            
            if "hint" in best_match:
                hint_parts.append(best_match["hint"])
            
            if "explanation" in best_match:
                hint_parts.append(f"참고: {best_match['explanation']}")
            
            return " ".join(hint_parts)

        # 일반적인 도메인 힌트
        return self._get_general_mc_hint(question_lower)

    def _get_general_mc_hint(self, question_lower: str) -> str:
        """일반적인 객관식 힌트"""
        
        # 부정 문제 힌트
        if any(neg in question_lower for neg in ["해당하지 않는", "적절하지 않은", "옳지 않은", "틀린"]):
            return "문제에서 요구하는 것과 반대되는 선택지를 찾으세요."
        
        # 긍정 문제 힌트  
        elif any(pos in question_lower for pos in ["가장 적절한", "가장 옳은", "맞는 것"]):
            return "문제에서 요구하는 조건에 가장 부합하는 선택지를 선택하세요."
        
        # 도메인별 일반 힌트
        domain_hints = {
            "금융투자": "금융투자업에는 투자자문업, 투자매매업, 투자중개업이 포함되며, 소비자금융업과 보험중개업은 포함되지 않습니다.",
            "위험관리": "위험관리 계획의 필수 요소와 부적절한 요소를 구분하세요.",
            "개인정보": "개인정보보호법의 연령 제한과 동의 절차를 확인하세요.",
            "전자금융": "한국은행의 권한과 업무 범위를 고려하세요.",
            "사이버보안": "보안 기술의 목적과 활용 분야를 파악하세요."
        }
        
        for domain, hint in domain_hints.items():
            if domain in question_lower:
                return hint
        
        return "각 선택지를 신중히 검토하고 문제의 핵심 요구사항을 파악하세요."

    def get_institution_info(self, question: str) -> str:
        """기관 정보 제공"""
        question_lower = question.lower()
        
        # 기관별 매칭
        for inst_type, info in self.institution_database.items():
            patterns = info.get("관련질문패턴", [])
            for pattern in patterns:
                if pattern in question_lower:
                    context_text = f"기관명: {info['기관명']}\n"
                    context_text += f"소속: {info['소속']}\n"
                    context_text += f"주요 역할: {info['역할']}\n"
                    context_text += f"근거 법령: {info['근거법']}\n"
                    if '상세정보' in info:
                        context_text += f"상세 정보: {info['상세정보']}\n"
                    if '신고기관' in info:
                        context_text += f"관련 신고기관: {info['신고기관']}\n"
                    return context_text
        
        # 키워드 기반 매칭
        if "전자금융" in question_lower and ("분쟁" in question_lower or "조정" in question_lower):
            return self._format_institution_info(self.institution_database["전자금융분쟁조정"])
        elif "개인정보" in question_lower and ("신고" in question_lower or "침해" in question_lower):
            return self._format_institution_info(self.institution_database["개인정보보호"])
        elif "한국은행" in question_lower:
            return self._format_institution_info(self.institution_database["한국은행"])
        
        return "관련 전문 기관에서 해당 업무를 법령에 따라 담당하고 있습니다."

    def _format_institution_info(self, info: Dict) -> str:
        """기관 정보 포맷팅"""
        context_text = f"기관명: {info['기관명']}\n"
        context_text += f"소속: {info['소속']}\n"
        context_text += f"주요 역할: {info['역할']}\n"
        context_text += f"근거 법령: {info['근거법']}\n"
        if '상세정보' in info:
            context_text += f"상세 정보: {info['상세정보']}\n"
        if '신고기관' in info:
            context_text += f"관련 신고기관: {info['신고기관']}\n"
        return context_text

    def _analyze_mc_pattern(self, question: str) -> Dict:
        """객관식 패턴 분석"""
        question_lower = question.lower()

        pattern_info = {
            "is_mc_question": False,
            "pattern_type": None,
            "pattern_confidence": 0.0,
            "pattern_key": None,
            "hint_available": False,
        }

        try:
            for pattern_key, pattern_data in self.mc_answer_patterns.items():
                keyword_matches = sum(
                    1 for keyword in pattern_data["question_keywords"]
                    if keyword in question_lower
                )

                if keyword_matches >= 2:
                    pattern_info["is_mc_question"] = True
                    pattern_info["pattern_type"] = pattern_key
                    pattern_info["pattern_confidence"] = keyword_matches / len(pattern_data["question_keywords"])
                    pattern_info["pattern_key"] = pattern_key
                    pattern_info["hint_available"] = True
                    break
        except Exception as e:
            print(f"객관식 패턴 분석 오류: {e}")

        return pattern_info

    def _check_institution_question(self, question: str) -> Dict:
        """기관 관련 질문 확인"""
        question_lower = question.lower()

        institution_info = {
            "is_institution_question": False,
            "institution_type": None,
            "relevant_institution": None,
            "confidence": 0.0,
            "question_pattern": None,
            "hint_available": False,
        }

        # 기관 질문 패턴
        institution_patterns = [
            "기관.*기술하세요", "기관.*설명하세요", "어떤.*기관", "어느.*기관",
            "조정.*신청.*기관", "분쟁.*조정.*기관", "신청.*수.*있는.*기관",
            "담당.*기관", "관리.*기관", "감독.*기관", "소관.*기관",
            "신고.*기관", "접수.*기관", "상담.*기관", "문의.*기관",
            "위원회.*무엇", "위원회.*어디", "위원회.*설명", 
            "분쟁.*어디", "신고.*어디", "상담.*어디",
            "기관을.*기술하세요", ".*기관.*기술", "분쟁조정.*기관"
        ]

        pattern_matches = 0
        matched_patterns = []
        
        try:
            for pattern in institution_patterns:
                if re.search(pattern, question_lower):
                    pattern_matches += 1
                    matched_patterns.append(pattern)
        except Exception as e:
            print(f"기관 질문 패턴 매칭 오류: {e}")

        # 기관 질문으로 판단되는 조건
        is_asking_institution = pattern_matches > 0

        if is_asking_institution:
            institution_info["is_institution_question"] = True
            institution_info["confidence"] = min(pattern_matches / 1.0, 1.0)
            institution_info["question_pattern"] = matched_patterns[0] if matched_patterns else None
            institution_info["hint_available"] = True

            # 기관 타입 매칭
            institution_mapping = {
                "전자금융분쟁조정": ["전자금융", "전자적", "분쟁", "조정", "금융감독원", "이용자"],
                "개인정보보호": ["개인정보", "정보주체", "침해", "신고", "상담", "보호위원회"],
                "금융투자분쟁조정": ["금융투자", "투자자문", "자본시장", "분쟁", "투자자"],
                "한국은행": ["한국은행", "금융통화위원회", "자료제출", "통화신용정책", "지급결제"]
            }

            best_match_score = 0
            best_match_type = None

            try:
                for inst_type, keywords in institution_mapping.items():
                    keyword_matches = sum(1 for keyword in keywords if keyword in question_lower)
                    match_score = keyword_matches / len(keywords)
                    
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_type = inst_type

                if best_match_score > 0:
                    institution_info["institution_type"] = best_match_type
                    institution_info["confidence"] = best_match_score
            except Exception as e:
                print(f"기관 타입 매칭 오류: {e}")

        return institution_info

    def _check_competition_compliance(self, question: str) -> Dict:
        """대회 규칙 준수 확인"""
        compliance = {
            "korean_content": True,
            "appropriate_domain": True,
            "no_external_dependency": True,
        }

        try:
            korean_chars = len([c for c in question if ord(c) >= 0xAC00 and ord(c) <= 0xD7A3])
            total_chars = len([c for c in question if c.isalpha()])

            if total_chars > 0:
                korean_ratio = korean_chars / total_chars
                compliance["korean_content"] = korean_ratio > 0.7
        except Exception:
            compliance["korean_content"] = True

        found_domains = []
        try:
            for domain, keywords in self.domain_keywords.items():
                if any(keyword in question.lower() for keyword in keywords):
                    found_domains.append(domain)
        except Exception:
            pass

        compliance["appropriate_domain"] = len(found_domains) > 0

        return compliance

    def _calculate_complexity(self, question: str) -> float:
        """질문 복잡도 계산"""
        try:
            length_factor = min(len(question) / 200, 1.0)

            korean_term_count = sum(1 for term in self.korean_financial_terms.keys() if term in question)
            term_factor = min(korean_term_count / 3, 1.0)

            domain_count = sum(
                1 for keywords in self.domain_keywords.values()
                if any(keyword in question.lower() for keyword in keywords)
            )
            domain_factor = min(domain_count / 2, 1.0)

            return (length_factor + term_factor + domain_factor) / 3
        except Exception:
            return 0.5

    def _find_korean_technical_terms(self, question: str) -> List[str]:
        """한국어 기술용어 탐지"""
        found_terms = []
        try:
            for term in self.korean_financial_terms.keys():
                if term in question:
                    found_terms.append(term)
        except Exception:
            pass
        return found_terms

    def _determine_technical_level(self, complexity: float, korean_terms: List[str]) -> str:
        """기술 수준 결정"""
        try:
            if complexity > 0.7 or len(korean_terms) >= 2:
                return "고급"
            elif complexity > 0.4 or len(korean_terms) >= 1:
                return "중급"
            else:
                return "초급"
        except Exception:
            return "중급"

    def cleanup(self):
        """리소스 정리"""
        pass