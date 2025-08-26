# data_processor.py

import re
import unicodedata
from typing import Dict, List, Tuple
from config import KOREAN_REQUIREMENTS


class DataProcessor:
    """데이터 처리"""

    def __init__(self):
        self._initialize_data()
        self.korean_requirements = KOREAN_REQUIREMENTS.copy()

    def _initialize_data(self):
        """데이터 초기화"""
        
        # 객관식 패턴
        self.mc_patterns = [
            r"1\s+[가-힣\w].*\n2\s+[가-힣\w].*\n3\s+[가-힣\w]",
            r"①.*②.*③.*④.*⑤",
            r"1\s+[가-힣].*2\s+[가-힣].*3\s+[가-힣].*4\s+[가-힣].*5\s+[가-힣]",
            r"1\s+.*2\s+.*3\s+.*4\s+.*5\s+",
            r"1\.\s*.*2\.\s*.*3\.\s*.*4\.\s*.*5\.",
            r"1\)\s*.*2\)\s*.*3\)\s*.*4\)\s*.*5\)"
        ]

        # 객관식 키워드
        self.mc_keywords = [
            "해당하지.*않는.*것", "적절하지.*않는.*것", "옳지.*않는.*것", "틀린.*것",
            "맞는.*것", "옳은.*것", "적절한.*것", "올바른.*것", "가장.*적절한.*것",
            "가장.*옳은.*것", "구분.*해당하지.*않는.*것", "다음.*중.*것은", "다음.*중.*것",
            "다음.*보기.*중", "무엇인가\\?$", "어떤.*것인가\\?$", "몇.*개인가\\?$",
            "가장.*중요한.*것", "우선적으로.*고려.*것", "필수.*사항.*것"
        ]

        # 질문 의도 패턴
        self.question_intent_patterns = {
            "기관_묻기": [
                "기관.*기술하세요", "기관.*설명하세요", "기관.*서술하세요", "기관.*무엇",
                "어떤.*기관", "어느.*기관", "기관.*어디", "분쟁조정.*신청.*기관",
                "조정.*신청.*기관", "분쟁.*조정.*기관", "신청.*수.*있는.*기관",
                "분쟁.*해결.*기관", "조정.*담당.*기관", "감독.*기관", "관리.*기관",
                "담당.*기관", "주관.*기관", "소관.*기관", "신고.*기관", "접수.*기관",
                "상담.*기관", "문의.*기관", "위원회.*무엇", "위원회.*어디", "위원회.*설명",
                "전자금융.*분쟁.*기관", "전자금융.*조정.*기관", "개인정보.*신고.*기관",
                "개인정보.*보호.*기관", "개인정보.*침해.*기관", "기관을.*기술하세요",
                ".*기관.*기술", "분쟁조정.*기관", "신청할.*수.*있는.*기관",
                "한국은행.*자료제출", "금융감독원.*분쟁조정", "보호위원회.*업무"
            ],
            "특징_묻기": [
                "특징.*설명하세요", "특징.*기술하세요", "특징.*서술하세요", "어떤.*특징",
                "주요.*특징", "특징.*무엇", "성격.*설명", "성질.*설명", "속성.*설명",
                "특성.*설명", "특성.*무엇", "성격.*무엇", "특성.*기술", "속성.*기술",
                "기반.*원격제어.*악성코드.*특징", "트로이.*특징", "RAT.*특징",
                ".*특징.*설명하세요", ".*특징.*기술하세요", "트로이.*목마.*특징",
                "원격제어.*악성코드.*특징", "악성코드.*특징", "딥페이크.*특징",
                "SBOM.*특징", "암호화.*특징", "접근통제.*특징", "디지털.*지갑.*특징",
                "SMTP.*역할", "프로토콜.*역할", "3대.*요소", "보안.*목표", "보안.*위협",
                "우려되는.*주요.*보안.*위협", "주요.*보안.*위협"
            ],
            "지표_묻기": [
                "지표.*설명하세요", "탐지.*지표", "주요.*지표", "어떤.*지표", "지표.*무엇",
                "징후.*설명", "신호.*설명", "패턴.*설명", "행동.*패턴", "활동.*패턴",
                "모니터링.*지표", "관찰.*지표", "식별.*지표", "발견.*방법", "탐지.*방법",
                "주요.*탐지.*지표", "악성코드.*탐지.*지표", "원격제어.*탐지.*지표",
                ".*탐지.*지표.*설명하세요", ".*지표.*설명하세요", "주요.*탐지.*지표",
                "탐지.*지표.*무엇", "보안.*이벤트.*지표", "침입.*탐지.*지표"
            ],
            "방안_묻기": [
                "방안.*기술하세요", "방안.*설명하세요", "대응.*방안", "해결.*방안",
                "관리.*방안", "어떤.*방안", "대책.*설명", "조치.*방안", "처리.*방안",
                ".*방안", "예방.*방안", "보완.*방안", "딥페이크.*대응.*방안",
                "금융권.*대응.*방안", "악용.*대비.*방안", "보안.*방안", "위험.*관리.*방안",
                "개인정보.*보호.*방안", "전자금융.*보안.*방안", "선제적.*대응.*방안"
            ],
            "절차_묻기": [
                "절차.*설명하세요", "절차.*기술하세요", "어떤.*절차", "처리.*절차",
                "진행.*절차", "수행.*절차", "실행.*절차", "과정.*설명", "단계.*설명",
                "프로세스.*설명", "동의.*절차", "신고.*절차", "조정.*절차",
                "어떻게.*수행해야"
            ],
            "조치_묻기": [
                "조치.*설명하세요", "조치.*기술하세요", "어떤.*조치", "보안.*조치",
                "대응.*조치", "예방.*조치", ".*조치", "보완.*조치", "기술적.*조치"
            ],
            "법령_묻기": [
                "법령.*설명", "법률.*설명", "규정.*설명", "조항.*설명", "규칙.*설명",
                "기준.*설명", "법적.*근거", "관련.*법", "적용.*법", "법에.*따라"
            ],
            "정의_묻기": [
                "정의.*설명", "개념.*설명", "의미.*설명", "뜻.*설명", "무엇.*의미",
                "무엇.*뜻", "용어.*설명", "개념.*무엇", "정의.*무엇"
            ],
            "원칙_묻기": [
                "원칙.*설명", "원칙.*기술", "기본.*원칙", "원칙.*무엇", "적용.*원칙",
                "준수.*원칙", "관리.*원칙", "보안.*원칙"
            ],
            "비율_묻기": [
                "비율.*얼마", "기준.*비율", "비율.*무엇", "몇.*퍼센트", "어느.*정도",
                "기준.*얼마"
            ]
        }

        # 주관식 패턴
        self.subj_patterns = [
            "설명하세요", "기술하세요", "서술하세요", "작성하세요", "무엇인가요",
            "어떻게.*해야.*하며", "방안을.*기술", "대응.*방안", "특징.*다음과.*같",
            "탐지.*지표", "행동.*패턴", "분석하여.*제시", "조치.*사항", "제시하시오",
            "논하시오", "답하시오", "특징과.*주요.*탐지.*지표를.*설명하세요",
            "기관을.*기술하세요", "대응.*방안을.*기술하세요", "절차를.*설명하세요"
        ]

        # 한국어 복구 설정
        self.korean_recovery_config = {
            "broken_unicode_chars": {
                "\\u1100": "", "\\u1101": "", "\\u1102": "", "\\u1103": "", "\\u1104": "",
                "\\u1105": "", "\\u1106": "", "\\u1107": "", "\\u1108": "", "\\u1109": "",
                "\\u110A": "", "\\u110B": "", "\\u110C": "", "\\u110D": "", "\\u110E": "",
                "\\u110F": "", "\\u1110": "", "\\u1111": "", "\\u1112": "", "\\u1161": "",
            },
            "spaced_korean_fixes": {
                "작 로": "으로", "렴": "련", "니 터": "니터", "지 속": "지속", "모 니": "모니",
                "체 계": "체계", "관 리": "관리", "법 령": "법령", "규 정": "규정", "조 치": "조치",
                "절 차": "절차", "대 응": "대응", "방 안": "방안", "기 관": "기관", "위 원": "위원",
                "감 독": "감독", "전 자": "전자", "금 융": "금융", "개 인": "개인", "정 보": "정보",
                "보 호": "보호", "관 련": "관련", "필 요": "필요", "중 요": "중요", "주 요": "주요",
            }
        }

        # 한국어 품질 패턴
        self.korean_quality_patterns = [
            {"pattern": r"([가-힣])\s+(은|는|이|가|을|를|에|의|와|과|로|으로)\s+", "replacement": r"\1\2 "},
            {"pattern": r"([가-힣])\s+(다|요|함|니다|습니다)\s*\.", "replacement": r"\1\2."},
            {"pattern": r"([가-힣])\s*$", "replacement": r"\1."},
            {"pattern": r"\.+", "replacement": "."},
            {"pattern": r"\s*\.\s*", "replacement": ". "},
            {"pattern": r"\s+", "replacement": " "},
            {"pattern": r"\(\s*\)", "replacement": ""},
            {"pattern": r"\(\s*\)\s*[가-힣]{1,3}", "replacement": ""},
            {"pattern": r"[.,!?]{3,}", "replacement": "."},
            {"pattern": r"\s+[.,!?]\s+", "replacement": ". "}
        ]

        # 도메인 키워드
        self.domain_keywords = {
            "개인정보보호": [
                "개인정보", "정보주체", "개인정보보호법", "민감정보", "고유식별정보",
                "수집", "이용", "제공", "파기", "동의", "법정대리인", "아동", "처리",
                "개인정보처리방침", "열람권", "정정삭제권", "처리정지권", "손해배상",
                "개인정보보호위원회", "개인정보영향평가", "개인정보관리체계",
                "개인정보처리시스템", "개인정보보호책임자", "개인정보취급자",
                "개인정보침해신고센터", "PIMS", "관리체계", "정책",
                "만 14세", "미만 아동", "중요한 요소", "경영진", "최고책임자",
                "자원", "내부 감사", "처리 위탁", "수탁자", "위탁자",
                "개인정보 처리 현황", "처리방침", "고지", "공개", "통지", "접근 권한"
            ],
            "전자금융": [
                "전자금융", "전자적", "접근매체", "전자금융거래법", "전자서명",
                "전자인증", "공인인증서", "분쟁조정", "전자지급수단", "전자화폐",
                "금융감독원", "한국은행", "전자금융업", "전자금융분쟁조정위원회",
                "전자금융거래", "전자금융업무", "전자금융서비스", "전자금융거래기록",
                "이용자", "금융통화위원회", "자료제출", "통화신용정책", "지급결제제도",
                "요청", "요구", "경우", "보안", "통계조사", "경영", "운영",
                "전자금융업자", "보안시스템", "거래", "손해", "과실",
                "접근매체", "부정거래", "이용", "승인", "기록", "정보보호", "예산",
                "정보기술부문", "인력", "전자금융감독규정"
            ],
            "사이버보안": [
                "트로이", "악성코드", "멀웨어", "바이러스", "피싱", "스미싱", "랜섬웨어",
                "해킹", "딥페이크", "원격제어", "RAT", "원격접근", "봇넷", "백도어",
                "루트킷", "취약점", "제로데이", "사회공학", "APT", "DDoS", "침입탐지",
                "침입방지", "보안관제", "SBOM", "소프트웨어 구성 요소", "Trojan",
                "원격제어 악성코드", "탐지 지표", "보안 위협", "특징", "주요 탐지",
                "금융권", "활용", "이유", "적절한", "소프트웨어", "접근 제어",
                "투명성", "다양성", "공급망 보안", "행동 분석", "네트워크 모니터링",
                "실시간 탐지", "SIEM", "보안 이벤트", "위협", "디지털 지갑", "보안 위협"
            ],
            "정보보안": [
                "정보보안", "보안관리", "ISMS", "보안정책", "접근통제", "암호화",
                "방화벽", "침입탐지", "침입방지시스템", "IDS", "IPS", "보안관제",
                "로그관리", "백업", "복구", "재해복구", "BCP", "정보보안관리체계",
                "정보보호", "관리체계", "정책", "최고책임자", "경영진",
                "자원", "내부 감사", "절차", "복구 절차", "비상연락체계",
                "개인정보 파기", "복구 목표시간", "옳지 않은", "고려", "요소",
                "보안 감사", "취약점 점검", "보안 교육", "사고 대응", "보안 운영",
                "정보보호", "3대 요소", "보안 목표", "SMTP", "프로토콜", "보안상 주요 역할"
            ],
            "금융투자": [
                "금융투자업", "투자자문업", "투자매매업", "투자중개업", "소비자금융업",
                "보험중개업", "자본시장법", "집합투자업", "신탁업", "펀드", "파생상품",
                "투자자보호", "적합성원칙", "설명의무", "금융산업", "구분",
                "해당하지 않는", "금융산업의 이해", "내부통제", "리스크 관리",
                "투자 권유", "투자 위험", "고객 적합성"
            ],
            "위험관리": [
                "위험관리", "위험평가", "위험대응", "위험수용", "리스크", "내부통제",
                "컴플라이언스", "위험식별", "위험분석", "위험모니터링", "위험회피",
                "위험전가", "위험감소", "잔여위험", "위험성향", "위험 관리 계획",
                "수행인력", "위험 대응 전략", "재해 복구", "복구 절차", "비상연락체계",
                "복구 목표시간", "계획 수립", "고려", "요소", "적절하지 않은", "대상", "기간",
                "위험 허용 수준", "위험 보고", "위험 통제", "위험 지표"
            ],
            "정보통신": [
                "정보통신시설", "집적된 정보통신시설", "정보통신서비스", "과학기술정보통신부장관",
                "보고", "중단", "발생", "일시", "장소", "원인", "법적 책임", "피해내용", "응급조치"
            ]
        }

        self._setup_korean_recovery_mappings()

    def _setup_korean_recovery_mappings(self):
        """한국어 복구 매핑 설정"""
        self.korean_recovery_mapping = {}

        # 깨진 유니코드 문자 처리
        for broken, replacement in self.korean_recovery_config["broken_unicode_chars"].items():
            try:
                actual_char = broken.encode().decode("unicode_escape")
                self.korean_recovery_mapping[actual_char] = replacement
            except Exception:
                continue

        # 기타 매핑 추가
        self.korean_recovery_mapping.update(self.korean_recovery_config["spaced_korean_fixes"])

    def extract_choice_range(self, question: str) -> Tuple[str, int]:
        """선택지 범위 추출"""
        # 먼저 주관식 패턴으로 확실히 판별
        question_type = self.analyze_question_type(question)
        
        if question_type == "subjective":
            return "subjective", 0

        # 객관식인 경우 선택지 개수 파악
        lines = question.split("\n")
        choice_numbers = []
        choice_contents = {}

        # 선택지 패턴 검사
        for line in lines:
            line = line.strip()
            
            # 기본 숫자 패턴
            match = re.match(r"^(\d+)\s+(.+)", line)
            if match:
                try:
                    num = int(match.group(1))
                    content = match.group(2).strip()
                    if 1 <= num <= 5 and len(content) > 0:
                        choice_numbers.append(num)
                        choice_contents[num] = content
                except ValueError:
                    continue
            
            # 다른 패턴들도 검사
            for pattern in [r"^(\d+)\)\s*(.+)", r"^(\d+)\.\s*(.+)"]:
                match = re.match(pattern, line)
                if match:
                    try:
                        num = int(match.group(1))
                        content = match.group(2).strip()
                        if 1 <= num <= 5 and len(content) > 0:
                            choice_numbers.append(num)
                            choice_contents[num] = content
                    except ValueError:
                        continue

        if choice_numbers:
            choice_numbers.sort()
            max_choice = max(choice_numbers)
            min_choice = min(choice_numbers)

            # 연속성 검사
            expected_count = max_choice - min_choice + 1
            if (len(set(choice_numbers)) == expected_count and 
                min_choice == 1 and max_choice >= 3):
                
                # 선택지 내용 품질 검사
                valid_choices = 0
                for num in choice_numbers:
                    if num in choice_contents:
                        content = choice_contents[num]
                        if len(content) >= 3 and not content.isdigit():
                            valid_choices += 1
                
                if valid_choices >= 3:
                    return "multiple_choice", max_choice

        # 패턴 기반 검사
        for i in range(5, 2, -1):
            pattern_parts = [f"{j}\\s+[가-힣\\w]{{2,}}" for j in range(1, i + 1)]
            pattern = ".*".join(pattern_parts)
            try:
                if re.search(pattern, question, re.DOTALL):
                    return "multiple_choice", i
            except Exception:
                continue

        return "multiple_choice", 5

    def analyze_question_type(self, question: str) -> str:
        """질문 유형 분석"""
        question = question.strip()

        # 주관식 패턴 확실한 검사
        for pattern in self.subj_patterns:
            try:
                if re.search(pattern, question, re.IGNORECASE):
                    return "subjective"
            except Exception:
                continue

        # 명확한 주관식 키워드
        subjective_keywords = ["설명하세요", "기술하세요", "서술하세요", "작성하세요"]
        if any(keyword in question for keyword in subjective_keywords):
            return "subjective"

        # 선택지 패턴 검사 
        try:
            choice_patterns = [
                r"\n(\d+)\s+[가-힣\w]{2,}",
                r"\n(\d+)\)\s*[가-힣\w]{2,}",
                r"\n(\d+)\.\s*[가-힣\w]{2,}"
            ]
            
            for pattern in choice_patterns:
                choice_matches = re.findall(pattern, question)
                if len(choice_matches) >= 3:
                    choice_nums = []
                    try:
                        choice_nums = [int(match) for match in choice_matches]
                        choice_nums.sort()
                        if (choice_nums[0] == 1 and 
                            len(choice_nums) == choice_nums[-1] and 
                            choice_nums[-1] <= 5):
                            return "multiple_choice"
                    except ValueError:
                        continue
        except Exception:
            pass

        # 키워드 기반 검사
        mc_score = 0
        for pattern in self.mc_keywords:
            try:
                if re.search(pattern, question, re.IGNORECASE):
                    mc_score += 1
                    if mc_score >= 1 and len(re.findall(r'\b[1-5]\s+[가-힣\w]', question)) >= 3:
                        return "multiple_choice"
            except Exception:
                continue

        # 패턴 기반 검사
        for pattern in self.mc_patterns:
            try:
                if re.search(pattern, question, re.DOTALL | re.MULTILINE):
                    return "multiple_choice"
            except Exception:
                continue

        # 문제 끝 패턴 기반 판별
        try:
            if (len(question) < 500 and 
                re.search(r"것은\?|것\?|것은\s*$|무엇인가\?", question) and 
                len(re.findall(r"\b[1-5]\s+[가-힣\w]{2,}", question)) >= 4):
                return "multiple_choice"
        except Exception:
            pass

        return "subjective"

    def extract_domain(self, question: str) -> str:
        """도메인 추출"""
        question_lower = question.lower()
        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    # 핵심 키워드에 더 높은 가중치
                    if keyword in [
                        "개인정보보호법", "전자금융거래법", "자본시장법", "ISMS",
                        "트로이", "RAT", "원격제어", "SBOM", "딥페이크",
                        "전자금융분쟁조정위원회", "개인정보보호위원회", 
                        "만 14세", "법정대리인", "위험 관리", "금융투자업",
                        "재해 복구", "접근통제", "암호화", "디지털 지갑",
                        "SMTP", "정보보호", "3대 요소", "정보통신시설"
                    ]:
                        score += 8
                    elif keyword in [
                        "개인정보", "전자금융", "사이버보안", "정보보안", 
                        "금융투자", "위험관리"
                    ]:
                        score += 5
                    elif keyword in [
                        "보안", "관리", "정책", "법령", "규정", "조치"
                    ]:
                        score += 2
                    else:
                        score += 1

            if score > 0:
                domain_scores[domain] = score

        if not domain_scores:
            return "일반"

        # 최고 점수 도메인 선택
        detected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]

        # 도메인별 추가 검증
        if detected_domain == "사이버보안":
            cybersec_keywords = ["트로이", "악성코드", "RAT", "원격제어", "딥페이크", "SBOM", "보안", "탐지", "디지털 지갑"]
            if any(keyword in question_lower for keyword in cybersec_keywords):
                return "사이버보안"
        elif detected_domain == "개인정보보호":
            privacy_keywords = ["개인정보", "정보주체", "만 14세", "법정대리인", "PIMS", "동의", "처리", "접근 권한"]
            if any(keyword in question_lower for keyword in privacy_keywords):
                return "개인정보보호"
        elif detected_domain == "전자금융":
            finance_keywords = ["전자금융", "분쟁조정", "한국은행", "금융감독원", "자료제출", "통화신용정책", "정보기술부문", "예산"]
            if any(keyword in question_lower for keyword in finance_keywords):
                return "전자금융"
        elif detected_domain == "정보보안":
            infosec_keywords = ["정보보안", "ISMS", "재해복구", "접근통제", "암호화", "보안정책", "3대 요소", "SMTP"]
            if any(keyword in question_lower for keyword in infosec_keywords):
                return "정보보안"
        elif detected_domain == "위험관리":
            risk_keywords = ["위험관리", "위험평가", "위험대응", "위험수용", "내부통제"]
            if any(keyword in question_lower for keyword in risk_keywords):
                return "위험관리"
        elif detected_domain == "금융투자":
            investment_keywords = ["금융투자업", "투자자문", "투자매매", "투자중개", "소비자금융", "보험중개"]
            if any(keyword in question_lower for keyword in investment_keywords):
                return "금융투자"
        elif detected_domain == "정보통신":
            it_keywords = ["정보통신시설", "정보통신서비스", "과학기술정보통신부장관"]
            if any(keyword in question_lower for keyword in it_keywords):
                return "정보통신"

        return detected_domain

    def analyze_question_intent(self, question: str) -> Dict:
        """질문 의도 분석"""
        question_lower = question.lower()

        intent_analysis = {
            "primary_intent": "일반",
            "intent_confidence": 0.0,
            "detected_patterns": [],
            "answer_type_required": "설명형",
            "secondary_intents": [],
            "context_hints": [],
            "quality_risk": False,
        }

        # 의도별 점수 계산
        intent_scores = {}

        for intent_type, patterns in self.question_intent_patterns.items():
            score = 0
            matched_patterns = []

            for pattern in patterns:
                try:
                    matches = re.findall(pattern, question, re.IGNORECASE)
                    if matches:
                        if len(matches) > 1:
                            score += 4.0
                        elif ".*" in pattern and len(pattern) > 15:
                            score += 3.0
                        else:
                            score += 2.0
                        matched_patterns.append(pattern)
                except Exception:
                    continue

            # 키워드 보너스
            keyword_bonuses = {
                "기관_묻기": ["기관", "위원회", "담당", "업무", "어디", "누가", "신청할", "분쟁조정", "한국은행", "금융감독원", "보호위원회"],
                "특징_묻기": ["특징", "특성", "성질", "속성", "어떤", "트로이", "RAT", "원격제어", "악성코드", "딥페이크", "보안 위협", "주요", "역할"],
                "지표_묻기": ["지표", "징후", "탐지", "모니터링", "신호", "주요", "패턴", "행동", "활동"],
                "방안_묻기": ["방안", "대책", "해결", "대응", "어떻게", "조치", "예방", "보안"],
                "절차_묻기": ["절차", "과정", "단계", "순서", "프로세스", "동의", "신고", "어떻게", "수행"],
                "조치_묻기": ["조치", "대응", "예방", "보안", "기술적"],
                "원칙_묣기": ["원칙", "기본", "준수", "적용", "관리"],
                "비율_묻기": ["비율", "얼마", "기준", "퍼센트", "정도"]
            }

            if intent_type in keyword_bonuses:
                keyword_matches = sum(
                    1 for keyword in keyword_bonuses[intent_type] if keyword in question_lower
                )
                if keyword_matches > 0:
                    if keyword_matches >= 2:
                        score += keyword_matches * 2.0
                    else:
                        score += keyword_matches * 1.5

            # 특정 조합 추가 점수
            if intent_type == "특징_묻기":
                if "트로이" in question_lower and ("특징" in question_lower or "RAT" in question_lower):
                    score += 6.0
                if "원격제어" in question_lower and "특징" in question_lower:
                    score += 6.0
                if "악성코드" in question_lower and "특징" in question_lower:
                    score += 4.0
                if "딥페이크" in question_lower and "특징" in question_lower:
                    score += 5.0
                if "디지털 지갑" in question_lower and "보안 위협" in question_lower:
                    score += 6.0
                if "SMTP" in question_lower and "역할" in question_lower:
                    score += 6.0
                if "3대 요소" in question_lower:
                    score += 6.0
                    
            elif intent_type == "지표_묻기":
                if "탐지" in question_lower and "지표" in question_lower:
                    score += 6.0
                if "주요" in question_lower and ("지표" in question_lower or "탐지" in question_lower):
                    score += 4.0
                    
            elif intent_type == "기관_묻기":
                if "분쟁조정" in question_lower and ("신청" in question_lower or "기관" in question_lower):
                    score += 7.0
                if "전자금융" in question_lower and "기관" in question_lower:
                    score += 6.0
                if "개인정보" in question_lower and ("신고" in question_lower or "상담" in question_lower):
                    score += 6.0
                if "한국은행" in question_lower and "자료제출" in question_lower:
                    score += 5.0

            elif intent_type == "방안_묻기":
                if "딥페이크" in question_lower and "대응" in question_lower:
                    score += 6.0
                if "보안" in question_lower and "방안" in question_lower:
                    score += 4.0

            elif intent_type == "절차_묻기":
                if "어떻게" in question_lower and "수행" in question_lower:
                    score += 5.0

            elif intent_type == "비율_묻기":
                if "비율" in question_lower and "얼마" in question_lower:
                    score += 6.0

            if score > 0:
                intent_scores[intent_type] = {"score": score, "patterns": matched_patterns}

        # 최고 점수 의도 선택
        if intent_scores:
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1]["score"], reverse=True)
            best_intent = sorted_intents[0]

            intent_analysis["primary_intent"] = best_intent[0]
            intent_analysis["intent_confidence"] = min(best_intent[1]["score"] / 8.0, 1.0)
            intent_analysis["detected_patterns"] = best_intent[1]["patterns"]

            if len(sorted_intents) > 1:
                intent_analysis["secondary_intents"] = [
                    {"intent": intent, "score": data["score"]}
                    for intent, data in sorted_intents[1:3]
                ]

            # 답변 유형 결정
            primary = best_intent[0]
            if "기관" in primary:
                intent_analysis["answer_type_required"] = "기관명"
                intent_analysis["context_hints"].append("구체적인 기관명과 법적 근거")
            elif "특징" in primary:
                intent_analysis["answer_type_required"] = "특징설명"
                intent_analysis["context_hints"].append("기술적 특성과 동작 방식")
            elif "지표" in primary:
                intent_analysis["answer_type_required"] = "지표나열"
                intent_analysis["context_hints"].append("구체적 탐지 지표와 모니터링 방법")
            elif "방안" in primary:
                intent_analysis["answer_type_required"] = "방안제시"
                intent_analysis["context_hints"].append("단계별 실행방안과 구체적 조치")
            elif "절차" in primary:
                intent_analysis["answer_type_required"] = "절차설명"
                intent_analysis["context_hints"].append("법적 절차와 단계별 과정")
            elif "조치" in primary:
                intent_analysis["answer_type_required"] = "조치설명"
                intent_analysis["context_hints"].append("기술적 보안조치 내용")
            elif "원칙" in primary:
                intent_analysis["answer_type_required"] = "원칙설명"
                intent_analysis["context_hints"].append("기본 원칙과 적용 방법")
            elif "비율" in primary:
                intent_analysis["answer_type_required"] = "수치설명"
                intent_analysis["context_hints"].append("정확한 수치와 법적 근거")

        # 추가 문맥 분석
        self._add_context_analysis(question, intent_analysis)

        return intent_analysis

    def _add_context_analysis(self, question: str, intent_analysis: Dict):
        """문맥 분석 추가"""
        question_lower = question.lower()

        # 복합 질문 처리
        if "특징" in question_lower and "지표" in question_lower:
            intent_analysis["context_hints"].append("특징과 탐지지표 복합 질문")
            intent_analysis["answer_type_required"] = "복합설명"

        # 법적 근거 요구
        legal_keywords = ["법령", "법률", "규정", "조항", "법에", "근거", "조건"]
        if any(keyword in question_lower for keyword in legal_keywords):
            intent_analysis["context_hints"].append("법적 근거와 조항 포함")

    def detect_english_response(self, text: str) -> bool:
        """영어 답변 감지"""
        if not text:
            return False
        
        try:
            english_words = re.findall(r'\b[a-zA-Z]+\b', text)
            
            if len(english_words) > 15:
                return True
            
            english_sentences = re.findall(r'[A-Z][a-zA-Z\s,\.]{30,}', text)
            if len(english_sentences) > 1:
                return True
                
            english_terms = ['Relation', 'relevant', 'laws', 'regulations', 'Trojans', 'Remote', 'Access', 'Tools', 'RATs', 'malware', 'computer', 'systems', 'networks', 'subjected', 'various', 'legal', 'frameworks', 'jurisdictions', 'worldwide']
            english_term_count = sum(1 for term in english_terms if term in text)
            if english_term_count > 5:
                return True
                
            english_patterns = [
                r'\b[A-Z][a-z]+\s+to\s+[a-z]+',
                r'\b[A-Z][a-z]+\s+and\s+[A-Z][a-z]+',
                r'\b[A-Z][a-z]+\s+are\s+[a-z]+',
                r'\b[A-Z][a-z]+\s+in\s+[a-z]+',
                r'as forms of',
                r'that can',
                r'are subjected to',
                r'in different'
            ]
            
            english_pattern_count = 0
            for pattern in english_patterns:
                if re.search(pattern, text):
                    english_pattern_count += 1
                    
            if english_pattern_count > 2:
                return True
                    
            return False
            
        except Exception as e:
            print(f"영어 답변 감지 오류: {e}")
            return False

    def detect_critical_repetitive_patterns(self, text: str) -> bool:
        """문제 패턴 감지"""
        if not text or len(text) < 25:
            return False

        critical_patterns = [r"(.{1,3})\s*(\1\s*){12,}"]

        for pattern in critical_patterns:
            try:
                if re.search(pattern, text):
                    return True
            except Exception:
                continue

        # 단어 반복 검사
        words = text.split()
        if len(words) >= 12:
            for i in range(len(words) - 11):
                same_count = 0
                for j in range(i, min(i + 12, len(words))):
                    if words[i] == words[j]:
                        same_count += 1
                    else:
                        break

                if same_count >= 12 and len(words[i]) <= 5:
                    return True

        return False

    def remove_critical_repetitive_patterns(self, text: str) -> str:
        """문제 패턴 제거"""
        if not text:
            return ""

        words = text.split()
        cleaned_words = []
        i = 0
        
        while i < len(words):
            current_word = words[i]
            count = 1
            
            while i + count < len(words) and words[i + count] == current_word:
                count += 1

            if len(current_word) <= 2:
                cleaned_words.extend([current_word] * min(4, count))
            elif len(current_word) <= 5:
                cleaned_words.extend([current_word] * min(6, count))
            elif count >= 12:
                cleaned_words.extend([current_word] * min(6, count))
            else:
                cleaned_words.extend([current_word] * count)

            i += count

        text = " ".join(cleaned_words)
        
        try:
            text = re.sub(r"(.{3,15})\s*\1\s*\1\s*\1\s*\1\s*\1+", r"\1", text)
            text = re.sub(r"(.{1,5})\s*(\1\s*){10,}", r"\1", text)
            text = re.sub(r"\(\s*\)", "", text)
            text = re.sub(r"\s*\(\s*\)\s*", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        except Exception:
            pass

        return text

    def restore_korean_characters(self, text: str) -> str:
        """한국어 문자 복구"""
        if not text:
            return ""

        if self.detect_critical_repetitive_patterns(text):
            text = self.remove_critical_repetitive_patterns(text)

        try:
            text = unicodedata.normalize("NFC", text)
        except Exception:
            pass

        for broken, correct in self.korean_recovery_mapping.items():
            text = text.replace(broken, correct)

        try:
            text = re.sub(r"\(\s*\)", "", text)
            text = re.sub(r"[.,!?]{3,}", ".", text)
            text = re.sub(r"\s+[.,!?]\s+", ". ", text)
        except Exception:
            pass

        return text

    def clean_korean_text(self, text: str) -> str:
        """한국어 텍스트 정리"""
        if not text:
            return ""

        if self.detect_critical_repetitive_patterns(text):
            text = self.remove_critical_repetitive_patterns(text)
            if len(text) < 8:
                return "텍스트 정리 중 내용이 부족합니다."

        text = self.restore_korean_characters(text)
        text = self.fix_grammatical_structure(text)
        
        try:
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"[^\w\s가-힣.,!?()[\]\-]", " ", text)
        except Exception:
            pass

        try:
            english_chars = len(re.findall(r"[a-zA-Z]", text))
            total_chars = len(re.sub(r"[^\w가-힣]", "", text))
            if total_chars > 0 and english_chars / total_chars > 0.6:
                text = re.sub(r"[a-zA-Z]+", "", text)
        except Exception:
            pass

        try:
            text = re.sub(r"[\u4e00-\u9fff]", "", text)
            text = re.sub(r"[①②③④⑤➀➁➂➃➄]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
        except Exception:
            pass

        if self.detect_critical_repetitive_patterns(text):
            text = self.remove_critical_repetitive_patterns(text)
            if len(text) < 10:
                return "텍스트 정리 후 내용이 부족합니다."

        return text

    def fix_grammatical_structure(self, text: str) -> str:
        """문법 구조 수정"""
        if not text:
            return ""

        if self.detect_critical_repetitive_patterns(text):
            text = self.remove_critical_repetitive_patterns(text)

        # 문법 수정
        grammar_fixes = [
            (r"([가-힣])\s+(은|는|이|가|을|를|에|의|와|과|로|으로)\s+", r"\1\2 "),
            (r"([가-힣])\s+(다|요|함|니다|습니다)\s*\.", r"\1\2."),
            (r"([가-힣])\s*$", r"\1."),
            (r"\.+", "."),
            (r"\s*\.\s*", ". "),
            (r"\s*,\s*", ", "),
            (r"([가-힣])\s*$", r"\1."),
        ]

        for pattern, replacement in grammar_fixes:
            try:
                text = re.sub(pattern, replacement, text)
            except Exception:
                continue

        # 문장별 처리
        sentences = text.split(".")
        processed_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            if self.detect_critical_repetitive_patterns(sentence):
                continue

            # 긴 문장 분할
            if len(sentence) > 300:
                try:
                    parts = re.split(r"[,，]", sentence)
                    if len(parts) > 1:
                        for part in parts:
                            part = part.strip()
                            if len(part) > 8 and not self.detect_critical_repetitive_patterns(part):
                                processed_sentences.append(part)
                    else:
                        if not self.detect_critical_repetitive_patterns(sentence):
                            processed_sentences.append(sentence)
                except Exception:
                    processed_sentences.append(sentence)
            else:
                if not self.detect_critical_repetitive_patterns(sentence):
                    processed_sentences.append(sentence)

        if processed_sentences:
            result = ". ".join(processed_sentences)
        else:
            result = "관련 법령과 규정에 따라 체계적인 관리를 수행해야 합니다"

        if result and not result.endswith("."):
            result += "."

        return result

    def calculate_korean_ratio(self, text: str) -> float:
        """한국어 비율 계산"""
        if not text:
            return 0.0

        try:
            korean_chars = len(re.findall(r"[가-힣]", text))
            total_chars = len(re.sub(r"[^\w가-힣]", "", text))

            if total_chars == 0:
                return 0.0

            return korean_chars / total_chars
        except Exception:
            return 0.0

    def calculate_english_ratio(self, text: str) -> float:
        """영어 비율 계산"""
        if not text:
            return 0.0

        try:
            english_chars = len(re.findall(r"[a-zA-Z]", text))
            total_chars = len(re.sub(r"[^\w가-힣]", "", text))

            if total_chars == 0:
                return 0.0

            return english_chars / total_chars
        except Exception:
            return 0.0

    def validate_mc_answer_range(self, answer: str, max_choice: int) -> bool:
        """객관식 답변 범위 확인"""
        if not answer or not answer.isdigit():
            return False

        try:
            answer_num = int(answer)
            return 1 <= answer_num <= max_choice
        except ValueError:
            return False

    def validate_answer_intent_match(self, answer: str, question: str, intent_analysis: Dict) -> bool:
        """답변과 의도 매칭 검증"""
        if not answer or not intent_analysis:
            return True

        if self.detect_critical_repetitive_patterns(answer):
            return False
            
        if self.detect_english_response(answer):
            return False

        basic_quality_keywords = [
            "법", "규정", "관리", "조치", "체계", "시스템", "보안", "업무",
            "담당", "수행", "필요", "해야", "구축", "수립", "시행", "실시",
            "특징", "지표", "탐지", "기관", "위원회", "방안", "대응", "절차"
        ]
        
        return any(word in answer.lower() for word in basic_quality_keywords)

    def validate_korean_answer(self, answer: str, question_type: str, max_choice: int = 5, question: str = "") -> bool:
        """한국어 답변 검증"""
        if not answer:
            return False

        answer = str(answer).strip()

        if self.detect_critical_repetitive_patterns(answer):
            return False
            
        if self.detect_english_response(answer):
            return False

        if question_type == "multiple_choice":
            if not self.validate_mc_answer_range(answer, max_choice):
                return False
            return True

        else:
            clean_answer = self.clean_korean_text(answer)

            if self.detect_critical_repetitive_patterns(clean_answer):
                return False
                
            if self.detect_english_response(clean_answer):
                return False

            if len(clean_answer) < 5:
                return False

            korean_ratio = self.calculate_korean_ratio(clean_answer)
            if korean_ratio < 0.2:
                return False

            english_ratio = self.calculate_english_ratio(answer)
            if english_ratio > 0.5:
                return False

            try:
                korean_chars = len(re.findall(r"[가-힣]", clean_answer))
                if korean_chars < 3:
                    return False
            except Exception:
                return False

            meaningful_keywords = [
                "법", "규정", "조치", "관리", "보안", "방안", "절차", "기준",
                "정책", "체계", "시스템", "통제", "특징", "지표", "탐지", "대응",
                "기관", "위원회", "감독원", "업무", "담당", "수행", "필요", "해야",
                "구축", "수립", "시행", "실시", "있", "는", "다", "을", "를", "의", "에"
            ]
            
            if any(word in clean_answer for word in meaningful_keywords):
                return True

            if len(clean_answer) >= 10:
                return True

            return False

    def analyze_question_difficulty(self, question: str) -> str:
        """질문 난이도 분석"""
        question_lower = question.lower()

        technical_terms = [
            "isms", "pims", "sbom", "원격제어", "침입탐지", "트로이", "멀웨어",
            "랜섬웨어", "딥페이크", "피싱", "접근매체", "전자서명", "rat",
            "개인정보보호법", "자본시장법", "전자금융거래법", "원격접근", "탐지지표",
            "apt", "ddos", "ids", "ips", "bcp", "drp", "isms-p",
            "분쟁조정", "금융투자업", "위험관리", "재해복구", "비상연락체계",
            "암호키관리", "최소권한원칙", "적합성원칙", "법정대리인", "디지털 지갑",
            "smtp", "정보통신시설"
        ]

        term_count = sum(1 for term in technical_terms if term in question_lower)
        length = len(question)
        choice_count = len(self.extract_choices(question))
        
        complexity_score = 0
        
        if term_count >= 3:
            complexity_score += 3
        elif term_count >= 2:
            complexity_score += 2
        elif term_count >= 1:
            complexity_score += 1
            
        if length > 400:
            complexity_score += 2
        elif length > 250:
            complexity_score += 1
            
        if choice_count >= 5:
            complexity_score += 1

        if "특징" in question_lower and "지표" in question_lower:
            complexity_score += 2
        if "방안" in question_lower and ("대응" in question_lower or "대비" in question_lower):
            complexity_score += 1

        if complexity_score >= 5:
            return "고급"
        elif complexity_score >= 2:
            return "중급"
        else:
            return "초급"

    def extract_choices(self, question: str) -> List[str]:
        """선택지 추출"""
        choices = []

        lines = question.split("\n")
        for line in lines:
            line = line.strip()
            
            patterns = [
                r"^(\d+)\s+(.+)",
                r"^(\d+)\)\s*(.+)",
                r"^(\d+)\.\s*(.+)"
            ]
            
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    try:
                        choice_num = int(match.group(1))
                        choice_content = match.group(2).strip()
                        if 1 <= choice_num <= 5 and len(choice_content) >= 2:
                            choices.append(choice_content)
                            break
                    except ValueError:
                        continue

        if len(choices) >= 3:
            return choices[:5]

        if not choices:
            fallback_patterns = [
                r"(\d+)\s+([^0-9\n]{3,}?)(?=\d+\s+|$)",
                r"(\d+)\)\s*([^0-9\n]{3,}?)(?=\d+\)|$)",
                r"(\d+)\.\s*([^0-9\n]{3,}?)(?=\d+\.|$)",
                r"[①②③④⑤]\s*([^①②③④⑤\n]{3,}?)(?=[①②③④⑤]|$)",
            ]

            for pattern in fallback_patterns:
                try:
                    matches = re.findall(pattern, question, re.MULTILINE | re.DOTALL)
                    if matches:
                        if isinstance(matches[0], tuple):
                            choices = [match[1].strip() for match in matches if len(match[1].strip()) >= 2]
                        else:
                            choices = [match.strip() for match in matches if len(match.strip()) >= 2]

                        if len(choices) >= 3:
                            break
                except Exception:
                    continue

        return choices[:5]

    def normalize_korean_answer(self, answer: str, question_type: str, max_choice: int = 5) -> str:
        """한국어 답변 정규화"""
        if not answer:
            return ""

        answer = str(answer).strip()
        
        if self.detect_english_response(answer):
            return ""

        if question_type == "multiple_choice":
            try:
                numbers = re.findall(r"[1-9]", answer)
                for num in numbers:
                    if 1 <= int(num) <= max_choice:
                        return num
            except Exception:
                pass
            return ""

        else:
            answer = self.clean_korean_text(answer)

            if self.detect_critical_repetitive_patterns(answer):
                if len(answer) > 30:
                    answer = self.remove_critical_repetitive_patterns(answer)
                    if len(answer) < 15:
                        return "답변 생성 중 반복 패턴이 감지되어 재생성이 필요합니다."
                else:
                    return "답변 생성 중 반복 패턴이 감지되어 재생성이 필요합니다."
                    
            if self.detect_english_response(answer):
                return ""

            if len(answer) < 5:
                return "답변 길이가 부족하여 생성에 실패했습니다."

            max_length = 650
            if len(answer) > max_length:
                try:
                    sentences = answer.split(". ")
                    valid_sentences = []

                    for sentence in sentences:
                        if not self.detect_critical_repetitive_patterns(sentence):
                            valid_sentences.append(sentence)
                        if len(valid_sentences) >= 6:
                            break

                    if valid_sentences:
                        answer = ". ".join(valid_sentences[:6])
                    else:
                        return "답변 정규화 중 유효한 문장을 찾을 수 없습니다."

                    if len(answer) > max_length:
                        answer = answer[:max_length]
                except Exception:
                    answer = answer[:max_length]

            if answer and not answer.endswith((".", "다", "요", "함")):
                if answer.endswith("니"):
                    answer += "다."
                elif answer.endswith("습"):
                    answer += "니다."
                else:
                    answer += "."

            return answer

    def validate_answer(self, answer: str, question_type: str, max_choice: int = 5, question: str = "") -> bool:
        """답변 검증"""
        return self.validate_korean_answer(answer, question_type, max_choice, question)

    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        return self.clean_korean_text(text)

    def normalize_answer(self, answer: str, question_type: str, max_choice: int = 5) -> str:
        """답변 정규화"""
        return self.normalize_korean_answer(answer, question_type, max_choice)

    def cleanup(self):
        """리소스 정리"""
        pass