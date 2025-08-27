# data_processor.py

import re
import unicodedata
from typing import Dict, List, Tuple
from config import KOREAN_REQUIREMENTS


class DataProcessor:
    """향상된 데이터 처리"""

    def __init__(self):
        self._initialize_enhanced_data()
        self.korean_requirements = KOREAN_REQUIREMENTS.copy()
        self.domain_keywords_expanded = self._expand_enhanced_domain_keywords()
        self.pattern_weights = self._initialize_pattern_weights()

    def _initialize_enhanced_data(self):
        """향상된 데이터 초기화"""
        
        # 더 정확한 객관식 패턴
        self.enhanced_mc_patterns = [
            r"1\s+[가-힣\w].*\n2\s+[가-힣\w].*\n3\s+[가-힣\w].*\n4\s+[가-힣\w].*\n5\s+[가-힣\w]",
            r"1\s+[가-힣\w].*2\s+[가-힣\w].*3\s+[가-힣\w].*4\s+[가-힣\w].*5\s+[가-힣\w]",
            r"1\)\s*[가-힣\w].*2\)\s*[가-힣\w].*3\)\s*[가-힣\w].*4\)\s*[가-힣\w].*5\)\s*[가-힣\w]",
            r"1\.\s*[가-힣\w].*2\.\s*[가-힣\w].*3\.\s*[가-힣\w].*4\.\s*[가-힣\w].*5\.\s*[가-힣\w]",
            r"①.*②.*③.*④.*⑤",
            r"\n1\s+.*\n2\s+.*\n3\s+.*\n4\s+.*\n5\s+",
            r"[1-5]\s+[가-힣]{2,}.*[1-5]\s+[가-힣]{2,}.*[1-5]\s+[가-힣]{2,}.*[1-5]\s+[가-힣]{2,}"
        ]

        # 더 정확한 주관식 패턴
        self.enhanced_subj_patterns = [
            r"설명하세요$", r"기술하세요$", r"서술하세요$", r"작성하세요$",
            r".*설명하세요\.$", r".*기술하세요\.$", r".*서술하세요\.$",
            r".*을\s*설명하세요", r".*를\s*설명하세요", r".*을\s*기술하세요", r".*를\s*기술하세요",
            r".*방안을\s*기술하세요", r".*절차를\s*설명하세요", r".*특징.*설명하세요",
            r".*지표.*설명하세요", r".*기관.*기술하세요", r".*대응.*방안.*기술하세요",
            r".*어떻게.*해야.*하며", r".*조치.*사항", r".*제시하시오", r".*논하시오", 
            r".*답하시오", r".*무엇인가요\?$", r".*어떤.*것인가요\?$", r".*몇.*개인가요\?$"
        ]

        # 향상된 객관식 키워드
        self.enhanced_mc_keywords = [
            "해당하지.*않는.*것", "적절하지.*않는.*것", "옳지.*않는.*것", "틀린.*것",
            "맞는.*것", "옳은.*것", "적절한.*것", "올바른.*것", "가장.*적절한.*것",
            "가장.*옳은.*것", "구분.*해당하지.*않는.*것", "다음.*중.*것은", "다음.*중.*것",
            "다음.*보기.*중", "무엇인가\\?$", "어떤.*것인가\\?$", "몇.*개인가\\?$",
            "가장.*중요한.*것", "우선적으로.*고려.*것", "필수.*사항.*것", "가장.*부적절한.*것",
            "잘못.*설명한.*것", "가장.*타당한.*것", "가장.*바람직한.*것", "반드시.*필요한.*것"
        ]

        # 향상된 질문 의도 패턴
        self.enhanced_question_intent_patterns = {
            "기관_묻기": [
                "기관.*기술하세요", "기관.*설명하세요", "기관.*서술하세요", "기관.*무엇",
                "어떤.*기관", "어느.*기관", "기관.*어디", "분쟁조정.*신청.*기관",
                "조정.*신청.*기관", "분쟁.*조정.*기관", "신청.*수.*있는.*기관",
                "분쟁.*해결.*기관", "조정.*담당.*기관", "감독.*기관", "관리.*기관",
                "담당.*기관", "주관.*기관", "소관.*기관", "신고.*기관", "접수.*기관",
                "상담.*기관", "문의.*기관", "위원회.*무엇", "위원회.*어디", "위원회.*설명",
                "전자금융.*분쟁.*기관", "전자금융.*조정.*기관", "개인정보.*신고.*기관",
                "개인정보.*보호.*기관", "개인정보.*침해.*기관", "기관을.*기술하세요",
                ".*기관.*기술", "분쟁조정.*기관", "신청할.*수.*있는.*기관"
            ],
            "특징_묻기": [
                "특징.*설명하세요", "특징.*기술하세요", "특징.*서술하세요", "어떤.*특징",
                "주요.*특징", "특징.*무엇", "성격.*설명", "성질.*설명", "속성.*설명",
                "특성.*설명", "특성.*무엇", "성격.*무엇", "특성.*기술", "속성.*기술",
                "기반.*원격제어.*악성코드.*특징", "트로이.*특징", "RAT.*특징",
                ".*특징.*설명하세요", ".*특징.*기술하세요", "트로이.*목마.*특징",
                "원격제어.*악성코드.*특징", "악성코드.*특징", "딥페이크.*특징",
                "SBOM.*특징", "암호화.*특징", "접근통제.*특징", "디지털.*지갑.*특징",
                "보안.*위협.*특징", "침입.*특징", "공격.*특징"
            ],
            "지표_묻기": [
                "지표.*설명하세요", "탐지.*지표", "주요.*지표", "어떤.*지표", "지표.*무엇",
                "징후.*설명", "신호.*설명", "패턴.*설명", "행동.*패턴", "활동.*패턴",
                "모니터링.*지표", "관찰.*지표", "식별.*지표", "발견.*방법", "탐지.*방법",
                "주요.*탐지.*지표", "악성코드.*탐지.*지표", "원격제어.*탐지.*지표",
                ".*탐지.*지표.*설명하세요", ".*지표.*설명하세요", "주요.*탐지.*지표",
                "탐지.*지표.*무엇", "보안.*이벤트.*지표", "침입.*탐지.*지표", "위협.*지표"
            ],
            "방안_묻기": [
                "방안.*기술하세요", "방안.*설명하세요", "대응.*방안", "해결.*방안",
                "관리.*방안", "어떤.*방안", "대책.*설명", "조치.*방안", "처리.*방안",
                ".*방안", "예방.*방안", "보완.*방안", "딥페이크.*대응.*방안",
                "금융권.*대응.*방안", "악용.*대비.*방안", "보안.*방안", "위험.*관리.*방안",
                "개인정보.*보호.*방안", "전자금융.*보안.*방안", "선제적.*대응.*방안",
                "종합적.*방안", "체계적.*방안", "효과적.*방안"
            ],
            "절차_묻기": [
                "절차.*설명하세요", "절차.*기술하세요", "어떤.*절차", "처리.*절차",
                "진행.*절차", "수행.*절차", "실행.*절차", "과정.*설명", "단계.*설명",
                "프로세스.*설명", "동의.*절차", "신고.*절차", "조정.*절차",
                "어떻게.*수행해야", "수행.*방법", "진행.*과정", "처리.*과정"
            ],
            "비율_묻기": [
                "비율.*얼마", "기준.*비율", "비율.*무엇", "몇.*퍼센트", "어느.*정도",
                "기준.*얼마", "비율은.*얼마인가요", "기준.*비율.*얼마", "정보기술부문.*비율",
                "예산.*비율", "인력.*비율", ".*%.*이상", "배정.*비율", "기준.*수치"
            ],
            "역할_묻기": [
                "역할.*설명", "역할.*기술", "주요.*역할", "보안상.*주요.*역할", "역할.*무엇",
                "기능.*설명", "업무.*설명", "담당.*업무", "수행.*역할", "프로토콜.*역할"
            ],
            "요소_묻기": [
                "요소.*설명", "중요한.*요소", "가장.*중요한.*요소", "고려.*요소", "구성.*요소",
                "3대.*요소", "핵심.*요소", "주요.*요소", "필수.*요소", "기본.*요소"
            ]
        }

        # 도메인별 확장 키워드
        self.enhanced_domain_keywords = {
            "개인정보보호": [
                "개인정보", "정보주체", "개인정보보호법", "민감정보", "고유식별정보",
                "수집", "이용", "제공", "파기", "동의", "법정대리인", "아동", "처리",
                "개인정보처리방침", "열람권", "정정삭제권", "처리정지권", "손해배상",
                "개인정보보호위원회", "개인정보영향평가", "개인정보관리체계",
                "개인정보처리시스템", "개인정보보호책임자", "개인정보취급자",
                "개인정보침해신고센터", "PIMS", "관리체계", "정책", "만 14세", "미만 아동",
                "중요한 요소", "경영진", "최고책임자", "자원", "내부 감사", "처리 위탁",
                "수탁자", "위탁자", "개인정보 처리 현황", "처리방침", "고지", "공개", 
                "통지", "접근 권한", "최소권한", "권한 검토", "정보보호 정책",
                "개인정보 관리체계", "수립 및 운영", "정책 수립", "최소권한 원칙"
            ],
            "전자금융": [
                "전자금융", "전자적", "접근매체", "전자금융거래법", "전자서명",
                "전자인증", "공인인증서", "분쟁조정", "전자지급수단", "전자화폐",
                "금융감독원", "한국은행", "전자금융업", "전자금융분쟁조정위원회",
                "전자금융거래", "전자금융업무", "전자금융서비스", "전자금융거래기록",
                "이용자", "금융통화위원회", "자료제출", "통화신용정책", "지급결제제도",
                "요청", "요구", "경우", "보안", "통계조사", "경영", "운영",
                "전자금융업자", "보안시스템", "거래", "손해", "과실", "접근매체",
                "부정거래", "이용", "승인", "기록", "정보보호", "예산", "정보기술부문",
                "인력", "전자금융감독규정", "비율", "5%", "7%", "16조", "배정"
            ],
            "사이버보안": [
                "트로이", "악성코드", "멀웨어", "바이러스", "피싱", "스미싱", "랜섬웨어",
                "해킹", "딥페이크", "원격제어", "RAT", "원격접근", "봇넷", "백도어",
                "루트킷", "취약점", "제로데이", "사회공학", "APT", "DDoS", "침입탐지",
                "침입방지", "보안관제", "SBOM", "소프트웨어 구성 요소", "Trojan",
                "원격제어 악성코드", "탐지 지표", "보안 위협", "특징", "주요 탐지",
                "금융권", "활용", "이유", "적절한", "소프트웨어", "접근 제어",
                "투명성", "다양성", "공급망 보안", "행동 분석", "네트워크 모니터링",
                "실시간 탐지", "SIEM", "보안 이벤트", "위협", "디지털 지갑", "보안 위협",
                "딥보이스", "탐지 기술", "선제적 대응", "다층 방어체계", "생체인증"
            ],
            "정보보안": [
                "정보보안", "보안관리", "ISMS", "보안정책", "접근통제", "암호화",
                "방화벽", "침입탐지", "침입방지시스템", "IDS", "IPS", "보안관제",
                "로그관리", "백업", "복구", "재해복구", "BCP", "정보보안관리체계",
                "정보보호", "관리체계", "정책", "최고책임자", "경영진", "자원",
                "내부 감사", "절차", "복구 절차", "비상연락체계", "개인정보 파기",
                "복구 목표시간", "옳지 않은", "고려", "요소", "보안 감사", "취약점 점검",
                "보안 교육", "사고 대응", "보안 운영", "정보보호", "3대 요소",
                "보안 목표", "SMTP", "프로토콜", "보안상 주요 역할", "기밀성", "무결성", "가용성"
            ],
            "금융투자": [
                "금융투자업", "투자자문업", "투자매매업", "투자중개업", "소비자금융업",
                "보험중개업", "자본시장법", "집합투자업", "신탁업", "펀드", "파생상품",
                "투자자보호", "적합성원칙", "설명의무", "금융산업", "구분",
                "해당하지 않는", "금융산업의 이해", "내부통제", "리스크 관리",
                "투자 권유", "투자 위험", "고객 적합성", "투자 판단", "투자 분석"
            ],
            "위험관리": [
                "위험관리", "위험평가", "위험대응", "위험수용", "리스크", "내부통제",
                "컴플라이언스", "위험식별", "위험분석", "위험모니터링", "위험회피",
                "위험전가", "위험감소", "잔여위험", "위험성향", "위험 관리 계획",
                "수행인력", "위험 대응 전략", "재해 복구", "복구 절차", "비상연락체계",
                "복구 목표시간", "계획 수립", "고려", "요소", "적절하지 않은", "대상", 
                "기간", "위험 허용 수준", "위험 보고", "위험 통제", "위험 지표",
                "위험 수용", "대응 전략 선정"
            ],
            "정보통신": [
                "정보통신시설", "집적된 정보통신시설", "정보통신서비스", "과학기술정보통신부장관",
                "보고", "중단", "발생", "일시", "장소", "원인", "법적 책임", "피해내용", 
                "응급조치", "정보통신기반 보호법", "중단 발생", "보고 사항", "옳지 않은"
            ]
        }

        self._setup_enhanced_korean_recovery_mappings()

    def _initialize_pattern_weights(self) -> Dict:
        """패턴 가중치 초기화"""
        return {
            "강한_객관식": 5.0,    # 명확한 객관식 패턴
            "약한_객관식": 2.0,    # 애매한 객관식 패턴
            "강한_주관식": 4.0,    # 명확한 주관식 패턴
            "약한_주관식": 1.5,    # 애매한 주관식 패턴
            "도메인_가중치": 1.5,  # 도메인 특화 키워드 가중치
            "길이_가중치": 0.1     # 질문 길이별 가중치
        }

    def _expand_enhanced_domain_keywords(self) -> Dict:
        """확장된 도메인 키워드"""
        expanded_keywords = {}
        
        try:
            # 추가 컨텍스트 키워드
            additional_context_keywords = {
                "개인정보보호": [
                    "개인정보 접근", "접근권한 관리", "최소권한 적용", "권한 검토 절차",
                    "정보보호 정책 수립", "관리체계 운영", "경영진 참여", "의사결정",
                    "자원 배정", "내부감사 시행", "처리현황 공개", "정보주체 권리",
                    "동의 철회", "열람 요구", "정정삭제 요구", "손해배상 청구"
                ],
                "전자금융": [
                    "전자금융거래 안전성", "보안조치 시행", "접근매체 관리", "거래기록 보존",
                    "분쟁조정 절차", "자료제출 요구", "통화정책 수행", "지급결제 운영",
                    "금융회사 예산관리", "정보보호 예산 배정", "기준 비율 준수"
                ],
                "사이버보안": [
                    "악성코드 탐지", "원격제어 차단", "트로이목마 분석", "딥페이크 탐지",
                    "SBOM 관리", "공급망 보안", "소프트웨어 투명성", "취약점 관리",
                    "보안위협 분석", "침입탐지 시스템", "행동패턴 분석", "실시간 모니터링"
                ],
                "정보보안": [
                    "정보보안관리체계", "보안정책 수립", "위험분석 수행", "보안대책 구현",
                    "재해복구 계획", "비상연락체계", "복구목표시간", "백업시스템",
                    "접근통제 관리", "암호화 적용", "보안감사 실시", "3대 보안요소"
                ],
                "위험관리": [
                    "위험식별 과정", "위험평가 기준", "대응전략 수립", "모니터링 체계",
                    "내부통제 시스템", "컴플라이언스 준수", "위험한도 설정"
                ],
                "금융투자": [
                    "투자자 보호", "적합성 원칙", "투자권유 절차", "설명의무 이행",
                    "내부통제 체계", "리스크 관리"
                ],
                "정보통신": [
                    "정보통신기반보호", "서비스 중단 대응", "보고 의무", "응급조치 절차",
                    "과학기술정보통신부 보고"
                ]
            }
            
            for domain, base_keywords in self.enhanced_domain_keywords.items():
                expanded = base_keywords.copy()
                if domain in additional_context_keywords:
                    expanded.extend(additional_context_keywords[domain])
                expanded_keywords[domain] = expanded
                
        except Exception as e:
            print(f"도메인 키워드 확장 실패: {e}")
            expanded_keywords = self.enhanced_domain_keywords
            
        return expanded_keywords

    def _setup_enhanced_korean_recovery_mappings(self):
        """향상된 한국어 복구 매핑 설정"""
        self.korean_recovery_mapping = {
            # 깨진 유니코드 문자
            "\\u1100": "", "\\u1101": "", "\\u1102": "", "\\u1103": "", "\\u1104": "",
            "\\u1105": "", "\\u1106": "", "\\u1107": "", "\\u1108": "", "\\u1109": "",
            # 공백이 들어간 한국어 수정
            "작 로": "으로", "렴": "련", "니 터": "니터", "지 속": "지속", "모 니": "모니",
            "체 계": "체계", "관 리": "관리", "법 령": "법령", "규 정": "규정", "조 치": "조치",
            "절 차": "절차", "대 응": "대응", "방 안": "방안", "기 관": "기관", "위 원": "위원",
            "감 독": "감독", "전 자": "전자", "금 융": "금융", "개 인": "개인", "정 보": "정보",
            "보 호": "보호", "관 련": "관련", "필 요": "필요", "중 요": "중요", "주 요": "주요",
            "모 니 터 링": "모니터링", "탐 지": "탐지", "발 견": "발견", "식 별": "식별",
            "분 석": "분석", "확 인": "확인", "점 검": "점검", "보 안": "보안", "위 험": "위험"
        }

    def extract_choice_range(self, question: str) -> Tuple[str, int]:
        """향상된 선택지 범위 추출"""
        
        # 1단계: 명확한 주관식 패턴 확인
        for pattern in self.enhanced_subj_patterns:
            try:
                if re.search(pattern, question, re.IGNORECASE):
                    return "subjective", 0
            except Exception:
                continue
        
        # 2단계: 명확한 객관식 패턴 확인
        for pattern in self.enhanced_mc_patterns:
            try:
                if re.search(pattern, question, re.DOTALL | re.MULTILINE):
                    return self._extract_enhanced_mc_choice_count(question)
            except Exception:
                continue

        # 3단계: 선택지 개수 기반 판단
        choice_count = self._count_enhanced_valid_choices(question)
        if choice_count >= 4:
            return "multiple_choice", choice_count

        # 4단계: 키워드 기반 분석 (가중치 적용)
        question_type_score = self._calculate_question_type_score(question)
        if question_type_score > 3.0:
            return "multiple_choice", max(choice_count, 5)
        elif question_type_score < -2.0:
            return "subjective", 0

        # 5단계: 최종 판단 (길이와 패턴 조합)
        return self._final_enhanced_type_determination(question, choice_count, question_type_score)

    def _calculate_question_type_score(self, question: str) -> float:
        """질문 유형 점수 계산 (가중치 적용)"""
        try:
            question_lower = question.lower()
            score = 0.0
            
            # 강한 객관식 패턴
            strong_mc_patterns = [
                "해당하지.*않는.*것", "적절하지.*않는.*것", "옳지.*않는.*것",
                "가장.*적절한.*것", "가장.*중요한.*것", "구분.*해당하지.*않는.*것"
            ]
            for pattern in strong_mc_patterns:
                if re.search(pattern, question_lower):
                    score += self.pattern_weights["강한_객관식"]
            
            # 약한 객관식 패턴
            weak_mc_patterns = [
                "다음.*중.*것", "무엇인가\\?", "어떤.*것인가\\?", "몇.*개인가\\?"
            ]
            for pattern in weak_mc_patterns:
                if re.search(pattern, question_lower):
                    score += self.pattern_weights["약한_객관식"]
            
            # 강한 주관식 패턴
            strong_subj_patterns = [
                "설명하세요", "기술하세요", "서술하세요", "방안을.*기술하세요"
            ]
            for pattern in strong_subj_patterns:
                if pattern in question_lower:
                    score -= self.pattern_weights["강한_주관식"]
            
            # 약한 주관식 패턴  
            weak_subj_patterns = [
                "어떻게.*해야", "무엇.*의미", "어떤.*특징", "주요.*지표"
            ]
            for pattern in weak_subj_patterns:
                if re.search(pattern, question_lower):
                    score -= self.pattern_weights["약한_주관식"]
            
            # 길이 가중치
            length_bonus = len(question) * self.pattern_weights["길이_가중치"]
            if len(question) > 300:  # 긴 질문은 주관식 경향
                score -= length_bonus * 0.5
            elif len(question) < 150:  # 짧은 질문은 객관식 경향
                score += length_bonus * 0.3
                
            return score
        except Exception:
            return 0.0

    def _extract_enhanced_mc_choice_count(self, question: str) -> Tuple[str, int]:
        """향상된 객관식 선택지 개수 추출"""
        lines = question.split("\n")
        choice_numbers = []
        
        # 다양한 패턴으로 선택지 추출
        patterns = [
            r"^(\d+)\s+(.+)",           # "1 선택지"
            r"^(\d+)\)\s*(.+)",         # "1) 선택지"
            r"^(\d+)\.\s*(.+)",         # "1. 선택지"
            r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*(.+)",  # 원문자
            r"^\s*(\d+)\s+([가-힣\w]{2,})"   # 공백 포함
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in patterns:
                try:
                    match = re.match(pattern, line)
                    if match:
                        if pattern.startswith("^[①②③④⑤"):
                            # 원문자 처리
                            choice_numbers.append(len(choice_numbers) + 1)
                        else:
                            num = int(match.group(1))
                            content = match.group(2).strip()
                            if 1 <= num <= 10 and len(content) >= 2:
                                choice_numbers.append(num)
                        break
                except (ValueError, IndexError):
                    continue

        if choice_numbers:
            choice_numbers.sort()
            max_choice = max(choice_numbers)
            min_choice = min(choice_numbers)
            
            # 연속성 확인
            if len(set(choice_numbers)) >= 3 and max_choice - min_choice + 1 == len(set(choice_numbers)):
                if max_choice <= 10:
                    return "multiple_choice", max_choice

        return "multiple_choice", 5

    def _count_enhanced_valid_choices(self, question: str) -> int:
        """향상된 유효한 선택지 개수 계산"""
        lines = question.split("\n")
        valid_choices = 0
        found_numbers = set()
        
        # 더 관대한 선택지 인식
        patterns = [
            r"^(\d+)\s+[가-힣\w]{1,}",      # 최소 1글자
            r"^(\d+)\)\s*[가-힣\w]{1,}",    # 최소 1글자
            r"^(\d+)\.\s*[가-힣\w]{1,}",    # 최소 1글자
            r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*[가-힣\w]{1,}",
            r"(\d+)\s+[가-힣]{2,}.*업"        # "업" 으로 끝나는 패턴
        ]

        for line in lines:
            line = line.strip()
            if len(line) < 3:  # 너무 짧은 라인 제외
                continue
                
            for pattern in patterns:
                try:
                    if pattern.startswith("^[①②③④⑤"):
                        if re.match(pattern, line):
                            valid_choices += 1
                            break
                    else:
                        match = re.match(pattern, line)
                        if match:
                            try:
                                num = int(match.group(1))
                                if 1 <= num <= 10 and num not in found_numbers:
                                    valid_choices += 1
                                    found_numbers.add(num)
                                    break
                            except (ValueError, IndexError):
                                continue
                except Exception:
                    continue

        return valid_choices

    def _final_enhanced_type_determination(self, question: str, choice_count: int, type_score: float) -> Tuple[str, int]:
        """향상된 최종 유형 결정"""
        question_lower = question.lower()
        
        # 도메인별 특화 판단
        domain = self.extract_domain(question)
        domain_context = self._get_domain_question_context(domain, question_lower)
        
        # 복합 점수 계산
        final_score = type_score + domain_context["type_bonus"]
        
        # 명확한 패턴이 있는 경우
        if abs(final_score) > 4.0:
            if final_score > 0:
                return "multiple_choice", max(choice_count, 5)
            else:
                return "subjective", 0
        
        # 선택지가 충분히 있는 경우
        if choice_count >= 4:
            return "multiple_choice", choice_count
        
        # 길이 기반 판단 (개선)
        if len(question) > 400:
            if any(word in question_lower for word in ["설명", "기술", "서술", "방안", "절차"]):
                return "subjective", 0
        elif len(question) < 200 and choice_count >= 2:
            return "multiple_choice", max(choice_count, 5)
        
        # 기본 결정
        if choice_count >= 2:
            return "multiple_choice", max(choice_count, 5)
        else:
            return "subjective", 0

    def _get_domain_question_context(self, domain: str, question_lower: str) -> Dict:
        """도메인별 질문 컨텍스트"""
        context = {"type_bonus": 0.0, "confidence": 0.5}
        
        domain_patterns = {
            "금융투자": {
                "mc_indicators": ["금융투자업", "구분", "해당하지", "투자자문업", "투자매매업"],
                "subj_indicators": []
            },
            "사이버보안": {
                "mc_indicators": ["SBOM", "활용", "딥페이크", "대응", "적절한"],
                "subj_indicators": ["트로이", "특징", "탐지", "지표", "설명하세요"]
            },
            "개인정보보호": {
                "mc_indicators": ["만 14세", "중요한", "요소", "경영진"],
                "subj_indicators": ["접근", "권한", "검토", "어떻게"]
            },
            "전자금융": {
                "mc_indicators": ["한국은행", "자료제출", "요구"],
                "subj_indicators": ["분쟁조정", "기관", "기술하세요", "비율", "얼마"]
            },
            "정보보안": {
                "mc_indicators": ["재해복구", "옳지", "3대요소"],
                "subj_indicators": ["SMTP", "역할", "설명하세요"]
            },
            "위험관리": {
                "mc_indicators": ["위험관리", "적절하지", "위험수용"],
                "subj_indicators": []
            },
            "정보통신": {
                "mc_indicators": ["정보통신서비스", "보고", "옳지"],
                "subj_indicators": []
            }
        }
        
        if domain in domain_patterns:
            pattern = domain_patterns[domain]
            
            mc_matches = sum(1 for indicator in pattern["mc_indicators"] 
                           if indicator in question_lower)
            subj_matches = sum(1 for indicator in pattern["subj_indicators"] 
                             if indicator in question_lower)
            
            if mc_matches >= 2:
                context["type_bonus"] = 2.0
                context["confidence"] = 0.8
            elif subj_matches >= 2:
                context["type_bonus"] = -2.0
                context["confidence"] = 0.8
            elif mc_matches > subj_matches:
                context["type_bonus"] = 1.0
                context["confidence"] = 0.6
            elif subj_matches > mc_matches:
                context["type_bonus"] = -1.0
                context["confidence"] = 0.6
        
        return context

    def extract_domain(self, question: str) -> str:
        """향상된 도메인 추출"""
        question_lower = question.lower()
        domain_scores = {}
        
        # 각 도메인별 점수 계산
        for domain, keywords in self.domain_keywords_expanded.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    # 키워드별 가중치 적용
                    weight = self._get_keyword_weight(keyword, domain)
                    score += weight
                    matched_keywords.append(keyword)
            
            if score > 0:
                # 도메인별 컨텍스트 보너스
                context_bonus = self._get_domain_context_bonus(domain, question_lower, matched_keywords)
                domain_scores[domain] = score + context_bonus

        if not domain_scores:
            return self._classify_unknown_domain_enhanced(question_lower)

        # 최고 점수 도메인 선택
        best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        
        # 신뢰도 검증
        if self._verify_domain_confidence(best_domain, question_lower, domain_scores[best_domain]):
            return best_domain
        else:
            return "일반"

    def _get_keyword_weight(self, keyword: str, domain: str) -> float:
        """키워드별 가중치 계산"""
        
        # 고가중치 키워드
        high_weight_keywords = {
            "개인정보보호": ["개인정보보호법", "만 14세", "법정대리인", "개인정보보호위원회", "PIMS"],
            "전자금융": ["전자금융거래법", "전자금융분쟁조정위원회", "한국은행", "통화신용정책", "전자금융감독규정"],
            "사이버보안": ["트로이", "RAT", "원격제어", "SBOM", "딥페이크", "디지털지갑"],
            "정보보안": ["정보보안관리체계", "ISMS", "재해복구", "3대요소", "SMTP"],
            "위험관리": ["위험관리", "위험평가", "위험대응", "내부통제"],
            "금융투자": ["금융투자업", "자본시장법", "투자자문업", "투자매매업"],
            "정보통신": ["정보통신기반보호법", "정보통신서비스", "과학기술정보통신부장관"]
        }
        
        # 중가중치 키워드
        medium_weight_keywords = {
            "개인정보보호": ["개인정보", "정보주체", "동의", "처리", "접근권한"],
            "전자금융": ["전자금융", "분쟁조정", "자료제출", "정보기술부문", "비율"],
            "사이버보안": ["악성코드", "보안위협", "탐지지표", "공급망보안"],
            "정보보안": ["정보보안", "접근통제", "암호화", "보안정책"],
            "위험관리": ["위험", "리스크", "관리", "대응"],
            "금융투자": ["투자", "금융", "증권", "투자자보호"],
            "정보통신": ["정보통신", "보고", "중단", "응급조치"]
        }
        
        if keyword in high_weight_keywords.get(domain, []):
            return 10.0
        elif keyword in medium_weight_keywords.get(domain, []):
            return 5.0
        elif len(keyword) >= 5:
            return 3.0
        elif len(keyword) >= 3:
            return 2.0
        else:
            return 1.0

    def _get_domain_context_bonus(self, domain: str, question_lower: str, matched_keywords: List[str]) -> float:
        """도메인별 컨텍스트 보너스"""
        
        context_patterns = {
            "개인정보보호": [
                (["만 14세", "법정대리인"], 5.0),
                (["접근", "권한", "검토"], 3.0),
                (["경영진", "중요한", "요소"], 4.0)
            ],
            "전자금융": [
                (["한국은행", "자료제출"], 5.0),
                (["분쟁조정", "신청", "기관"], 4.0),
                (["정보기술부문", "비율"], 6.0)
            ],
            "사이버보안": [
                (["트로이", "특징", "탐지"], 5.0),
                (["딥페이크", "대응", "방안"], 4.0),
                (["SBOM", "활용"], 4.0)
            ],
            "정보보안": [
                (["재해복구", "계획", "수립"], 4.0),
                (["3대", "요소"], 5.0),
                (["SMTP", "보안상", "역할"], 4.0)
            ]
        }
        
        bonus = 0.0
        if domain in context_patterns:
            for pattern_keywords, pattern_bonus in context_patterns[domain]:
                if all(keyword in " ".join(matched_keywords) or keyword in question_lower 
                      for keyword in pattern_keywords):
                    bonus += pattern_bonus
        
        return bonus

    def _verify_domain_confidence(self, domain: str, question_lower: str, score: float) -> bool:
        """도메인 분류 신뢰도 검증"""
        
        # 최소 점수 요구
        min_scores = {
            "개인정보보호": 8.0,
            "전자금융": 8.0,  
            "사이버보안": 7.0,
            "정보보안": 6.0,
            "위험관리": 5.0,
            "금융투자": 5.0,
            "정보통신": 6.0
        }
        
        min_required = min_scores.get(domain, 3.0)
        if score < min_required:
            return False
        
        # 도메인별 필수 키워드 확인
        required_keywords = {
            "개인정보보호": ["개인정보"],
            "전자금융": ["전자금융", "금융"],
            "사이버보안": ["보안", "악성코드", "트로이", "딥페이크", "SBOM"],
            "정보보안": ["정보보안", "보안"],
            "위험관리": ["위험"],
            "금융투자": ["금융", "투자"],
            "정보통신": ["정보통신"]
        }
        
        if domain in required_keywords:
            required = required_keywords[domain]
            if not any(keyword in question_lower for keyword in required):
                return False
        
        return True

    def _classify_unknown_domain_enhanced(self, question_lower: str) -> str:
        """향상된 미분류 도메인 분류"""
        
        # 기술 키워드 매핑 (확장)
        tech_mappings = {
            "암호화": "정보보안", "해시": "정보보안", "키": "정보보안", 
            "알고리즘": "정보보안", "인증": "정보보안", "감사": "정보보안",
            "로그": "정보보안", "방화벽": "정보보안", "네트워크": "정보보안",
            "서버": "정보보안", "시스템": "정보보안", "데이터베이스": "정보보안",
            "모니터링": "정보보안", "백업": "정보보안", "복구": "정보보안",
            "취약점": "사이버보안", "해킹": "사이버보안", "침입": "사이버보안",
            "바이러스": "사이버보안", "멀웨어": "사이버보안", "스미싱": "사이버보안",
            "투자": "금융투자", "펀드": "금융투자", "주식": "금융투자",
            "채권": "금융투자", "자본": "금융투자", "증권": "금융투자"
        }
        
        # 조합 키워드 확인
        for tech_keyword, domain in tech_mappings.items():
            if tech_keyword in question_lower:
                return domain
        
        # 문맥 기반 추론
        if any(word in question_lower for word in ["관리", "정책", "체계", "절차"]):
            if "보안" in question_lower:
                return "정보보안"
            elif "위험" in question_lower:
                return "위험관리"
        
        return "일반"

    def analyze_question_intent(self, question: str) -> Dict:
        """향상된 질문 의도 분석"""
        question_lower = question.lower()

        intent_analysis = {
            "primary_intent": "일반",
            "intent_confidence": 0.0,
            "detected_patterns": [],
            "answer_type_required": "설명형",
            "secondary_intents": [],
            "context_hints": [],
            "quality_risk": False,
            "complexity_level": "중급"
        }

        intent_scores = {}

        # 향상된 의도 분석
        for intent_type, patterns in self.enhanced_question_intent_patterns.items():
            score = 0
            matched_patterns = []

            for pattern in patterns:
                try:
                    matches = re.findall(pattern, question, re.IGNORECASE)
                    if matches:
                        pattern_weight = self._calculate_pattern_weight(pattern, intent_type)
                        score += pattern_weight
                        matched_patterns.append(pattern)
                except Exception:
                    continue

            # 도메인별 의도 가중치
            domain = self.extract_domain(question)
            domain_bonus = self._get_intent_domain_bonus(intent_type, domain, question_lower)
            score += domain_bonus

            if score > 0:
                intent_scores[intent_type] = {"score": score, "patterns": matched_patterns}

        # 결과 처리
        if intent_scores:
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1]["score"], reverse=True)
            best_intent = sorted_intents[0]

            intent_analysis["primary_intent"] = best_intent[0]
            intent_analysis["intent_confidence"] = min(best_intent[1]["score"] / 10.0, 1.0)
            intent_analysis["detected_patterns"] = best_intent[1]["patterns"]

            if len(sorted_intents) > 1:
                intent_analysis["secondary_intents"] = [
                    {"intent": intent, "score": data["score"]}
                    for intent, data in sorted_intents[1:3]
                ]

            # 답변 유형 결정
            intent_analysis.update(self._determine_answer_requirements(best_intent[0], question_lower))

        # 복잡도 분석
        intent_analysis["complexity_level"] = self.analyze_question_difficulty(question)
        
        return intent_analysis

    def _calculate_pattern_weight(self, pattern: str, intent_type: str) -> float:
        """패턴 가중치 계산"""
        
        # 패턴별 기본 가중치
        base_weights = {
            "기관_묻기": 3.0,
            "특징_묻기": 2.5,
            "지표_묻기": 2.5,
            "방안_묻기": 2.0,
            "절차_묻기": 2.0,
            "비율_묻기": 4.0,  # 높은 가중치
            "역할_묻기": 2.0,
            "요소_묻기": 2.0
        }
        
        base_weight = base_weights.get(intent_type, 1.0)
        
        # 패턴 복잡도에 따른 조정
        if len(pattern) > 20:
            base_weight *= 1.5
        elif ".*" in pattern:
            base_weight *= 1.2
        
        return base_weight

    def _get_intent_domain_bonus(self, intent_type: str, domain: str, question_lower: str) -> float:
        """의도별 도메인 보너스"""
        
        domain_intent_bonuses = {
            "사이버보안": {
                "특징_묻기": 3.0,  # 트로이, 딥페이크 특징
                "지표_묻기": 3.0,  # 탐지 지표
                "방안_묻기": 2.0   # 대응 방안
            },
            "전자금융": {
                "기관_묻기": 4.0,  # 분쟁조정위원회
                "비율_묻기": 5.0,  # 정보기술부문 비율
                "절차_묻기": 2.0
            },
            "개인정보보호": {
                "기관_묻기": 3.0,  # 개인정보보호위원회
                "절차_묻기": 3.0,  # 법정대리인 동의
                "요소_묻기": 4.0   # 경영진 참여
            },
            "정보보안": {
                "요소_묻기": 4.0,  # 3대 요소
                "역할_묻기": 3.0,  # SMTP 역할
                "방안_묻기": 2.0
            }
        }
        
        return domain_intent_bonuses.get(domain, {}).get(intent_type, 0.0)

    def _determine_answer_requirements(self, intent_type: str, question_lower: str) -> Dict:
        """답변 요구사항 결정"""
        
        requirements = {}
        
        if "기관" in intent_type:
            requirements.update({
                "answer_type_required": "기관명",
                "context_hints": ["구체적인 기관명과 법적 근거", "업무 범위와 연락처"]
            })
        elif "특징" in intent_type:
            requirements.update({
                "answer_type_required": "특징설명",
                "context_hints": ["기술적 특성과 동작 방식", "구체적 사례 포함"]
            })
        elif "지표" in intent_type:
            requirements.update({
                "answer_type_required": "지표나열",
                "context_hints": ["구체적 탐지 지표", "모니터링 방법", "행동 패턴"]
            })
        elif "방안" in intent_type:
            requirements.update({
                "answer_type_required": "방안제시", 
                "context_hints": ["단계별 실행방안", "구체적 조치사항", "효과적 대응방법"]
            })
        elif "비율" in intent_type:
            requirements.update({
                "answer_type_required": "수치설명",
                "context_hints": ["정확한 수치와 퍼센트", "법적 근거 조항", "예외 규정"]
            })
        elif "역할" in intent_type:
            requirements.update({
                "answer_type_required": "역할설명",
                "context_hints": ["주요 기능과 업무", "보안상 중요성", "적용 범위"]
            })
        elif "요소" in intent_type:
            requirements.update({
                "answer_type_required": "요소설명",
                "context_hints": ["핵심 구성요소", "상호 관계", "적용 원칙"]
            })
        
        # 복합 의도 처리
        if "특징" in question_lower and "지표" in question_lower:
            requirements.update({
                "answer_type_required": "복합설명",
                "context_hints": ["특징과 탐지지표 복합 질문", "기술적 특성과 모니터링 방법"]
            })
        
        return requirements

    def analyze_question_difficulty(self, question: str) -> str:
        """향상된 질문 난이도 분석"""
        question_lower = question.lower()

        # 확장된 기술 용어
        technical_terms = [
            "isms", "pims", "sbom", "원격제어", "침입탐지", "트로이", "멀웨어",
            "랜섬웨어", "딥페이크", "피싱", "접근매체", "전자서명", "rat",
            "개인정보보호법", "자본시장법", "전자금융거래법", "원격접근", "탐지지표",
            "apt", "ddos", "ids", "ips", "bcp", "drp", "isms-p", "pims",
            "분쟁조정", "금융투자업", "위험관리", "재해복구", "비상연락체계",
            "암호키관리", "최소권한원칙", "적합성원칙", "법정대리인", "디지털지갑",
            "smtp", "정보통신시설", "정보기술부문", "비율", "전자금융감독규정",
            "통화신용정책", "지급결제제도", "개인정보영향평가", "정보보안관리체계"
        ]

        # 난이도 점수 계산
        difficulty_score = 0
        
        # 기술 용어 개수
        term_count = sum(1 for term in technical_terms if term in question_lower)
        if term_count >= 4:
            difficulty_score += 4
        elif term_count >= 2:
            difficulty_score += 2
        elif term_count >= 1:
            difficulty_score += 1
            
        # 질문 길이
        length = len(question)
        if length > 500:
            difficulty_score += 3
        elif length > 300:
            difficulty_score += 2
        elif length > 150:
            difficulty_score += 1
            
        # 선택지 개수
        choice_count = len(self.extract_choices(question))
        if choice_count >= 5:
            difficulty_score += 1

        # 복합 질문 확인
        complexity_indicators = [
            ("특징", "지표"), ("방안", "절차"), ("기관", "업무"),
            ("비율", "기준"), ("역할", "기능"), ("요소", "원칙")
        ]
        
        for indicator1, indicator2 in complexity_indicators:
            if indicator1 in question_lower and indicator2 in question_lower:
                difficulty_score += 2

        # 법령 인용
        if any(law in question_lower for law in ["법", "조", "항", "규정", "지침"]):
            difficulty_score += 1

        # 최종 난이도 결정
        if difficulty_score >= 7:
            return "고급"
        elif difficulty_score >= 3:
            return "중급"
        else:
            return "초급"

    def detect_english_response(self, text: str) -> bool:
        """향상된 영어 답변 감지"""
        if not text:
            return False
        
        try:
            # 영어 단어 개수
            english_words = re.findall(r'\b[a-zA-Z]+\b', text)
            if len(english_words) > 20:  # 15 → 20 (더 관대)
                return True
            
            # 연속된 영어 문장
            english_sentences = re.findall(r'[A-Z][a-zA-Z\s,\.]{25,}', text)  # 30 → 25
            if len(english_sentences) > 1:
                return True
                
            # 특정 영어 패턴
            problematic_patterns = [
                r'Relation.*relevant.*laws',
                r'subjected.*various.*legal',
                r'computer.*systems.*networks',
                r'Remote.*Access.*Tools',
                r'malware.*systems.*control'
            ]
            
            pattern_matches = sum(1 for pattern in problematic_patterns if re.search(pattern, text))
            if pattern_matches > 2:  # 더 엄격
                return True
                
            # 전체 텍스트에서 영어 비율
            total_chars = len(re.sub(r'[^\w가-힣]', '', text))
            english_chars = len(re.findall(r'[a-zA-Z]', text))
            
            if total_chars > 0:
                english_ratio = english_chars / total_chars
                if english_ratio > 0.3:  # 30% 이상이면 영어 답변
                    return True
                    
            return False
            
        except Exception as e:
            print(f"영어 답변 감지 오류: {e}")
            return False

    def detect_critical_repetitive_patterns(self, text: str) -> bool:
        """향상된 반복 패턴 감지"""
        if not text or len(text) < 25:
            return False

        # 더 정교한 패턴들
        critical_patterns = [
            r"(.{1,3})\s*(\1\s*){6,}",        # 6회 이상 반복
            r"([가-힣]{1,2})\s*(\1\s*){5,}",  # 한글 5회 이상
            r"(\w+\s+)(\1){4,}",              # 단어 4회 이상
            r"(. ){5,}",                      # 점과 공백 반복
            r"(\([^)]*\)\s*){3,}"             # 괄호 패턴 반복
        ]

        for pattern in critical_patterns:
            try:
                if re.search(pattern, text):
                    return True
            except Exception:
                continue

        # 단어 수준 분석
        words = text.split()
        if len(words) >= 8:
            word_counts = {}
            for word in words:
                if len(word) <= 3:  # 짧은 단어만 체크
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # 과도한 반복 확인
            for word, count in word_counts.items():
                if count >= 6 and len(word) <= 2:  # 2글자 이하 6회 이상
                    return True
                elif count >= 8 and len(word) <= 3:  # 3글자 이하 8회 이상
                    return True

        return False

    def remove_critical_repetitive_patterns(self, text: str) -> str:
        """향상된 반복 패턴 제거"""
        if not text:
            return ""

        # 단어별 처리
        words = text.split()
        cleaned_words = []
        i = 0
        
        while i < len(words):
            current_word = words[i]
            count = 1
            
            # 연속 반복 카운트
            while i + count < len(words) and words[i + count] == current_word:
                count += 1

            # 적정 반복수 결정 (더 관대)
            if len(current_word) <= 1:
                max_repeat = 2
            elif len(current_word) <= 2:
                max_repeat = 3
            elif len(current_word) <= 4:
                max_repeat = 4
            else:
                max_repeat = 2

            cleaned_words.extend([current_word] * min(count, max_repeat))
            i += count

        text = " ".join(cleaned_words)
        
        # 정규식 기반 정리
        try:
            text = re.sub(r"(.{2,8})\s*\1{2,}", r"\1", text)  # 2회 이상 → 1회
            text = re.sub(r"(.)\s*\1{4,}", r"\1", text)      # 단일 문자 4회 이상 → 1회
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"[.,!?]{2,}", ".", text)          # 구두점 정리
        except Exception:
            pass

        return text

    def restore_korean_characters(self, text: str) -> str:
        """향상된 한국어 문자 복구"""
        if not text:
            return ""

        # 반복 패턴 먼저 제거
        if self.detect_critical_repetitive_patterns(text):
            text = self.remove_critical_repetitive_patterns(text)

        try:
            text = unicodedata.normalize("NFC", text)
        except Exception:
            pass

        # 복구 매핑 적용
        for broken, correct in self.korean_recovery_mapping.items():
            text = text.replace(broken, correct)

        # 품질 패턴 적용
        quality_patterns = [
            (r"\(\s*\)", ""),
            (r"[.,!?]{3,}", "."),
            (r"\s+[.,!?]\s+", ". "),
            (r"([가-힣])\s+(은|는|이|가|을|를|에|의|와|과|로|으로)\s+", r"\1\2 "),
            (r"\s+", " ")
        ]

        for pattern, replacement in quality_patterns:
            try:
                text = re.sub(pattern, replacement, text)
            except Exception:
                continue

        return text.strip()

    def calculate_korean_ratio(self, text: str) -> float:
        """한국어 비율 계산"""
        if not text:
            return 0.0

        try:
            korean_chars = len(re.findall(r"[가-힣]", text))
            total_chars = len(re.sub(r"[^\w가-힣]", "", text))
            return korean_chars / total_chars if total_chars > 0 else 0.0
        except Exception:
            return 0.0

    def calculate_english_ratio(self, text: str) -> float:
        """영어 비율 계산"""
        if not text:
            return 0.0

        try:
            english_chars = len(re.findall(r"[a-zA-Z]", text))
            total_chars = len(re.sub(r"[^\w가-힣]", "", text))
            return english_chars / total_chars if total_chars > 0 else 0.0
        except Exception:
            return 0.0

    def validate_korean_answer(self, answer: str, question_type: str, max_choice: int = 5, question: str = "") -> bool:
        """향상된 한국어 답변 검증"""
        if not answer:
            return False

        answer = str(answer).strip()

        # 반복 패턴 검사
        if self.detect_critical_repetitive_patterns(answer):
            return False
            
        # 영어 답변 검사
        if self.detect_english_response(answer):
            return False

        if question_type == "multiple_choice":
            try:
                answer_num = int(answer)
                return 1 <= answer_num <= max_choice
            except ValueError:
                return False
        else:
            # 주관식 검증 강화
            if len(answer) < 10:  # 최소 길이 증가
                return False

            korean_ratio = self.calculate_korean_ratio(answer)
            if korean_ratio < 0.3:  # 한국어 비율 요구
                return False

            english_ratio = self.calculate_english_ratio(answer)
            if english_ratio > 0.4:  # 영어 비율 제한
                return False

            # 한국어 문자 수 확인
            korean_chars = len(re.findall(r"[가-힣]", answer))
            if korean_chars < 5:
                return False

            # 의미있는 키워드 확인 (확장)
            meaningful_keywords = [
                "법", "규정", "조치", "관리", "보안", "방안", "절차", "기준", "정책", 
                "체계", "시스템", "통제", "특징", "지표", "탐지", "대응", "기관", 
                "위원회", "감독원", "업무", "담당", "수행", "필요", "해야", "구축", 
                "수립", "시행", "실시", "있", "는", "다", "을", "를", "의", "에",
                "비율", "퍼센트", "%", "이상", "5%", "7%", "인력", "예산", "원칙",
                "요소", "역할", "기능", "설명", "제공", "보장", "확보", "강화",
                "위험", "평가", "분석", "모니터링", "접근", "권한", "동의", "처리"
            ]
            
            keyword_count = sum(1 for word in meaningful_keywords if word in answer)
            return keyword_count >= 2  # 2개 이상 키워드 필요

    def extract_choices(self, question: str) -> List[str]:
        """향상된 선택지 추출"""
        choices = []

        lines = question.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 확장된 패턴들
            patterns = [
                r"^(\d+)\s+(.+)",
                r"^(\d+)\)\s*(.+)",
                r"^(\d+)\.\s*(.+)",
                r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*(.+)",
                r"^\s*(\d+)\s+([가-힣\w]{1,})"
            ]
            
            for pattern in patterns:
                try:
                    if pattern.startswith("^[①②③④⑤"):
                        match = re.match(pattern, line)
                        if match:
                            choice_content = match.group(1).strip()
                            if len(choice_content) >= 1:  # 최소 길이 완화
                                choices.append(choice_content)
                                break
                    else:
                        match = re.match(pattern, line)
                        if match:
                            try:
                                choice_num = int(match.group(1))
                                choice_content = match.group(2).strip()
                                if 1 <= choice_num <= 10 and len(choice_content) >= 1:
                                    choices.append(choice_content)
                                    break
                            except (ValueError, IndexError):
                                continue
                except Exception:
                    continue

        # 중복 제거 및 정렬
        if len(choices) >= 3:
            return list(dict.fromkeys(choices))[:10]  # 최대 10개

        return choices

    def normalize_korean_answer(self, answer: str, question_type: str, max_choice: int = 5) -> str:
        """향상된 한국어 답변 정규화"""
        if not answer:
            return ""

        answer = str(answer).strip()
        
        # 영어 답변 제거
        if self.detect_english_response(answer):
            return ""

        if question_type == "multiple_choice":
            # 숫자 추출 개선
            number_patterns = [
                r'정답[:：]?\s*(\d+)',
                r'답[:：]?\s*(\d+)',
                r'번호[:：]?\s*(\d+)',
                r'\b(\d+)\b'
            ]
            
            for pattern in number_patterns:
                matches = re.findall(pattern, answer)
                for match in matches:
                    num = int(match)
                    if 1 <= num <= max_choice:
                        return str(num)
            return ""
        else:
            # 주관식 정규화
            answer = self.restore_korean_characters(answer)

            # 반복 패턴 제거
            if self.detect_critical_repetitive_patterns(answer):
                answer = self.remove_critical_repetitive_patterns(answer)
                if len(answer) < 15:
                    return "답변 생성 중 반복 패턴이 감지되어 재생성이 필요합니다."
                    
            if len(answer) < 10:
                return "답변 길이가 부족하여 생성에 실패했습니다."

            # 길이 제한 (도메인별)
            max_lengths = {
                "사이버보안": 800,     # 증가
                "전자금융": 700,       # 증가
                "개인정보보호": 700,   # 증가
                "정보보안": 600,       # 증가
                "위험관리": 550,       # 증가
                "금융투자": 500,       # 증가
                "정보통신": 500        # 증가
            }
            
            domain = self.extract_domain(answer)  # 간단한 도메인 추정
            max_length = max_lengths.get(domain, 700)
            
            if len(answer) > max_length:
                sentences = re.split(r'[.!?]', answer)
                truncated_sentences = []
                current_length = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and current_length + len(sentence) + 2 <= max_length:
                        if not self.detect_critical_repetitive_patterns(sentence):
                            truncated_sentences.append(sentence)
                            current_length += len(sentence) + 2
                    else:
                        break
                
                if truncated_sentences:
                    answer = ". ".join(truncated_sentences)
                    if not answer.endswith('.'):
                        answer += "."
                else:
                    answer = answer[:max_length-3] + "..."

            # 마침표 처리 개선
            if answer and not answer.endswith((".", "다", "요", "함", "니다", "습니다")):
                if answer.endswith("니"):
                    answer += "다."
                elif answer.endswith("습"):
                    answer += "니다."
                elif answer.endswith(("해야", "필요", "있음", "중요", "가능")):
                    answer += "."
                else:
                    answer += "."

            return answer

    def validate_answer(self, answer: str, question_type: str, max_choice: int = 5, question: str = "") -> bool:
        """답변 검증"""
        return self.validate_korean_answer(answer, question_type, max_choice, question)

    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        return self.restore_korean_characters(text)

    def normalize_answer(self, answer: str, question_type: str, max_choice: int = 5) -> str:
        """답변 정규화"""
        return self.normalize_korean_answer(answer, question_type, max_choice)

    def cleanup(self):
        """리소스 정리"""
        pass