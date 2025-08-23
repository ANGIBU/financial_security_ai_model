# data_processor.py

import re
import unicodedata
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

from config import KOREAN_REQUIREMENTS


class SimpleDataProcessor:

    def __init__(self):
        self._initialize_integrated_data()

        self.korean_requirements = KOREAN_REQUIREMENTS.copy()
        self.korean_requirements["min_korean_ratio"] = 0.4
        self.korean_requirements["max_english_ratio"] = 0.3
        self.korean_requirements["min_length"] = 15

    def _initialize_integrated_data(self):
        """JSON 데이터를 코드 내부로 통합하여 초기화"""
        
        # mc_patterns 데이터
        self.mc_patterns = [
            "1\\s+[가-힣\\w].*\\n2\\s+[가-힣\\w].*\\n3\\s+[가-힣\\w]",
            "①.*②.*③.*④.*⑤",
            "1\\s+[가-힣].*2\\s+[가-힣].*3\\s+[가-힣].*4\\s+[가-힣].*5\\s+[가-힣]",
            "1\\s+.*2\\s+.*3\\s+.*4\\s+.*5\\s+",
            "1\\.\\s*.*2\\.\\s*.*3\\.\\s*.*4\\.\\s*.*5\\.",
            "1\\)\\s*.*2\\)\\s*.*3\\)\\s*.*4\\)\\s*.*5\\)"
        ]

        # mc_keywords 데이터
        self.mc_keywords = [
            "해당하지.*않는.*것",
            "적절하지.*않는.*것",
            "옳지.*않는.*것",
            "틀린.*것",
            "맞는.*것",
            "옳은.*것",
            "적절한.*것",
            "올바른.*것",
            "가장.*적절한.*것",
            "가장.*옳은.*것",
            "구분.*해당하지.*않는.*것",
            "다음.*중.*것은",
            "다음.*중.*것",
            "다음.*보기.*중",
            "무엇인가\\?$",
            "어떤.*것인가\\?$",
            "몇.*개인가\\?$"
        ]

        # question_intent_patterns 데이터
        self.question_intent_patterns = {
            "기관_묻기": [
                "기관.*기술하세요",
                "기관.*설명하세요",
                "기관.*서술하세요",
                "기관.*무엇",
                "어떤.*기관",
                "어느.*기관",
                "기관.*어디",
                "분쟁조정을.*신청할.*수.*있는.*기관",
                "조정.*신청.*기관",
                "분쟁.*조정.*기관",
                "신청.*수.*있는.*기관",
                "분쟁.*해결.*기관",
                "조정.*담당.*기관",
                "감독.*기관",
                "관리.*기관",
                "담당.*기관",
                "주관.*기관",
                "소관.*기관",
                "신고.*기관",
                "접수.*기관",
                "상담.*기관",
                "문의.*기관",
                "위원회.*무엇",
                "위원회.*어디",
                "위원회.*설명",
                "전자금융.*분쟁.*기관",
                "전자금융.*조정.*기관",
                "개인정보.*신고.*기관",
                "개인정보.*보호.*기관",
                "개인정보.*침해.*기관"
            ],
            "특징_묻기": [
                "특징.*설명하세요",
                "특징.*기술하세요",
                "특징.*서술하세요",
                "어떤.*특징",
                "주요.*특징",
                "특징.*무엇",
                "성격.*설명",
                "성질.*설명",
                "속성.*설명",
                "특성.*설명",
                "특성.*무엇",
                "성격.*무엇",
                "특성.*기술",
                "속성.*기술",
                "기반.*원격제어.*악성코드.*특징",
                "트로이.*특징",
                "RAT.*특징"
            ],
            "지표_묻기": [
                "지표.*설명하세요",
                "탐지.*지표",
                "주요.*지표",
                "어떤.*지표",
                "지표.*무엇",
                "징후.*설명",
                "신호.*설명",
                "패턴.*설명",
                "행동.*패턴",
                "활동.*패턴",
                "모니터링.*지표",
                "관찰.*지표",
                "식별.*지표",
                "발견.*방법",
                "탐지.*방법",
                "주요.*탐지.*지표",
                "악성코드.*탐지.*지표",
                "원격제어.*탐지.*지표"
            ],
            "방안_묻기": [
                "방안.*기술하세요",
                "방안.*설명하세요",
                "대응.*방안",
                "해결.*방안",
                "관리.*방안",
                "어떤.*방안",
                "대책.*설명",
                "조치.*방안",
                "처리.*방안",
                "개선.*방안",
                "예방.*방안",
                "보완.*방안",
                "강화.*방안",
                "딥페이크.*대응.*방안",
                "금융권.*대응.*방안",
                "악용.*대비.*방안"
            ],
            "절차_묻기": [
                "절차.*설명하세요",
                "절차.*기술하세요",
                "어떤.*절차",
                "처리.*절차",
                "진행.*절차",
                "수행.*절차",
                "실행.*절차",
                "과정.*설명",
                "단계.*설명",
                "프로세스.*설명"
            ],
            "조치_묻기": [
                "조치.*설명하세요",
                "조치.*기술하세요",
                "어떤.*조치",
                "보안.*조치",
                "대응.*조치",
                "예방.*조치",
                "개선.*조치",
                "강화.*조치",
                "보완.*조치"
            ],
            "법령_묻기": [
                "법령.*설명",
                "법률.*설명",
                "규정.*설명",
                "조항.*설명",
                "규칙.*설명",
                "기준.*설명",
                "법적.*근거",
                "관련.*법",
                "적용.*법"
            ],
            "정의_묻기": [
                "정의.*설명",
                "개념.*설명",
                "의미.*설명",
                "뜻.*설명",
                "무엇.*의미",
                "무엇.*뜻",
                "용어.*설명",
                "개념.*무엇"
            ]
        }

        # subj_patterns 데이터
        self.subj_patterns = [
            "설명하세요",
            "기술하세요",
            "서술하세요",
            "작성하세요",
            "무엇인가요",
            "어떻게.*해야.*하며",
            "방안을.*기술",
            "대응.*방안",
            "특징.*다음과.*같",
            "탐지.*지표",
            "행동.*패턴",
            "분석하여.*제시",
            "조치.*사항",
            "제시하시오",
            "논하시오",
            "답하시오",
            "특징과.*주요.*탐지.*지표를.*설명하세요",
            "기관을.*기술하세요",
            "대응.*방안을.*기술하세요"
        ]

        # korean_text_recovery 데이터
        self.korean_recovery_config = {
            "broken_unicode_chars": {
                "\\u1100": "",
                "\\u1101": "",
                "\\u1102": "",
                "\\u1103": "",
                "\\u1104": "",
                "\\u1105": "",
                "\\u1106": "",
                "\\u1107": "",
                "\\u1108": "",
                "\\u1109": "",
                "\\u110A": "",
                "\\u110B": "",
                "\\u110C": "",
                "\\u110D": "",
                "\\u110E": "",
                "\\u110F": "",
                "\\u1110": "",
                "\\u1111": "",
                "\\u1112": "",
                "\\u1161": "",
                "\\u1162": "",
                "\\u1163": "",
                "\\u1164": "",
                "\\u1165": "",
                "\\u1166": "",
                "\\u1167": "",
                "\\u1168": "",
                "\\u1169": "",
                "\\u116A": "",
                "\\u116B": "",
                "\\u116C": "",
                "\\u116D": "",
                "\\u116E": "",
                "\\u116F": "",
                "\\u1170": "",
                "\\u1171": "",
                "\\u1172": "",
                "\\u1173": "",
                "\\u1174": "",
                "\\u1175": "",
                "\\u11A8": "",
                "\\u11A9": "",
                "\\u11AA": "",
                "\\u11AB": "",
                "\\u11AC": "",
                "\\u11AD": "",
                "\\u11AE": "",
                "\\u11AF": "",
                "\\u11B0": "",
                "\\u11B1": "",
                "\\u11B2": "",
                "\\u11B3": "",
                "\\u11B4": "",
                "\\u11B5": "",
                "\\u11B6": "",
                "\\u11B7": "",
                "\\u11B8": "",
                "\\u11B9": "",
                "\\u11BA": "",
                "\\u11BB": "",
                "\\u11BC": "",
                "\\u11BD": "",
                "\\u11BE": "",
                "\\u11BF": "",
                "\\u11C0": "",
                "\\u11C1": "",
                "\\u11C2": ""
            },
            "japanese_katakana_removal": {
                "ト": "",
                "リ": "",
                "ス": "",
                "ン": "",
                "ー": "",
                "ィ": "",
                "ウ": "",
                "エ": "",
                "オ": "",
                "カ": "",
                "キ": "",
                "ク": "",
                "ケ": "",
                "コ": "",
                "サ": "",
                "シ": "",
                "セ": "",
                "ソ": "",
                "タ": "",
                "チ": "",
                "ツ": "",
                "テ": "",
                "ナ": "",
                "ニ": "",
                "ヌ": "",
                "ネ": "",
                "ノ": "",
                "ハ": "",
                "ヒ": "",
                "フ": "",
                "ヘ": "",
                "ホ": "",
                "マ": "",
                "ミ": "",
                "ム": "",
                "メ": "",
                "モ": "",
                "ヤ": "",
                "ユ": "",
                "ヨ": "",
                "ラ": "",
                "ル": "",
                "レ": "",
                "ロ": "",
                "ワ": "",
                "ヰ": "",
                "ヱ": "",
                "ヲ": ""
            },
            "broken_korean_patterns": {
                "어어지인": "",
                "선 어": "",
                "언 어": "",
                "순 어": "",
                "ᄒᆞᆫ": "",
                "ᄒᆞᆫ선": "",
                "어어지인가": "",
                "지인가": "",
                "가 시": "",
                "시 언": "",
                "언 어어": "",
                "지인)가": "",
                "순 어어": "",
                "지인.": ""
            },
            "spaced_korean_fixes": {
                "작 로": "으로",
                "렴": "련",
                "니 터": "니터",
                "지 속": "지속",
                "모 니": "모니",
                "체 계": "체계",
                "관 리": "관리",
                "법 령": "법령",
                "규 정": "규정",
                "조 치": "조치",
                "절 차": "절차",
                "대 응": "대응",
                "방 안": "방안",
                "기 관": "기관",
                "위 원": "위원",
                "감 독": "감독",
                "전 자": "전자",
                "금 융": "금융",
                "개 인": "개인",
                "정 보": "정보",
                "보 호": "보호",
                "관 련": "관련",
                "필 요": "필요",
                "중 요": "중요",
                "주 요": "주요",
                "시 스": "시스",
                "템": "템",
                "프 로": "프로",
                "세 스": "세스",
                "네 트": "네트",
                "워 크": "워크",
                "트 래": "트래",
                "픽": "픽",
                "파 일": "파일",
                "로 그": "로그",
                "연 결": "연결",
                "접 근": "접근",
                "권 한": "권한",
                "사 용": "사용",
                "계 정": "계정",
                "활 동": "활동",
                "패 턴": "패턴",
                "행 동": "행동",
                "모 니 터 링": "모니터링",
                "탐 지": "탐지",
                "발 견": "발견",
                "식 별": "식별",
                "분 석": "분석",
                "확 인": "확인",
                "점 검": "점검",
                "감 사": "감사",
                "보 안": "보안",
                "안 전": "안전",
                "위 험": "위험",
                "내 부": "내부",
                "외 부": "외부",
                "시 행": "시행",
                "실 시": "실시",
                "수 립": "수립",
                "구 축": "구축",
                "마 련": "마련",
                "준 비": "준비",
                "실 행": "실행",
                "진 행": "진행",
                "처 리": "처리",
                "관 찰": "관찰",
                "추 적": "추적",
                "기 록": "기록",
                "저 장": "저장",
                "백 업": "백업",
                "복 구": "복구",
                "복 원": "복원",
                "복 사": "복사",
                "이 동": "이동",
                "전 송": "전송",
                "수 신": "수신",
                "발 신": "발신",
                "전 달": "전달",
                "통 신": "통신",
                "연 락": "연락",
                "회 의": "회의",
                "논 의": "논의",
                "검 토": "검토",
                "평 가": "평가",
                "심 사": "심사",
                "심 의": "심의",
                "승 인": "승인",
                "허 가": "허가",
                "인 가": "인가",
                "등 록": "등록",
                "신 청": "신청",
                "요 청": "요청",
                "신 고": "신고",
                "보 고": "보고",
                "통 보": "통보",
                "고 지": "고지",
                "안 내": "안내",
                "지 침": "지침",
                "지 시": "지시",
                "명 령": "명령",
                "지 도": "지도",
                "교 육": "교육",
                "훈 련": "훈련",
                "연 수": "연수",
                "학 습": "학습",
                "습 득": "습득",
                "이 해": "이해",
                "파 악": "파악",
                "인 식": "인식",
                "감 지": "감지"
            },
            "common_korean_typos": {
                "윋": "융",
                "젂": "전",
                "엯": "연",
                "룐": "른",
                "겫": "결",
                "뷮": "분",
                "쟈": "저",
                "럭": "력",
                "솟": "솔",
                "쟣": "저",
                "뿣": "불",
                "뻙": "분",
                "걗": "것",
                "룊": "른",
                "믝": "미",
                "읶": "인",
                "멈": "멈",
                "솔": "솔",
                "랛": "란",
                "궗": "사",
                "쪗": "저",
                "롸": "로",
                "뿞": "분",
                "딞": "딘",
                "쭒": "주",
                "놟": "놓",
                "룍": "른",
                "쫒": "조",
                "놔": "놔"
            }
        }

        # korean_quality_patterns 데이터
        self.korean_quality_patterns = [
            {
                "pattern": r"([가-힣])\s+(은|는|이|가|을|를|에|의|와|과|로|으로)\s+",
                "replacement": r"\1\2 "
            },
            {
                "pattern": r"([가-힣])\s+(다|요|함|니다|습니다)\s*\.",
                "replacement": r"\1\2."
            },
            {
                "pattern": r"([가-힣])\s*$",
                "replacement": r"\1."
            },
            {
                "pattern": r"\.+",
                "replacement": "."
            },
            {
                "pattern": r"\s*\.\s*",
                "replacement": ". "
            },
            {
                "pattern": r"\s+",
                "replacement": " "
            },
            {
                "pattern": r"\(\s*\)",
                "replacement": ""
            },
            {
                "pattern": r"\(\s*\)\s*[가-힣]{1,3}",
                "replacement": ""
            },
            {
                "pattern": r"[.,!?]{3,}",
                "replacement": "."
            },
            {
                "pattern": r"\s+[.,!?]\s+",
                "replacement": ". "
            }
        ]

        # domain_keywords 데이터
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

        self._setup_korean_recovery_mappings()

        print("통합 데이터 초기화 완료")

    def _setup_korean_recovery_mappings(self):
        """한국어 복구 매핑 설정"""
        self.korean_recovery_mapping = {}

        for broken, replacement in self.korean_recovery_config["broken_unicode_chars"].items():
            try:
                actual_char = broken.encode().decode("unicode_escape")
                self.korean_recovery_mapping[actual_char] = replacement
            except:
                pass

        self.korean_recovery_mapping.update(
            self.korean_recovery_config["japanese_katakana_removal"]
        )

        self.korean_recovery_mapping.update(
            self.korean_recovery_config["broken_korean_patterns"]
        )

        self.korean_recovery_mapping.update(
            self.korean_recovery_config["spaced_korean_fixes"]
        )

        self.korean_recovery_mapping.update(
            self.korean_recovery_config["common_korean_typos"]
        )

        problematic_patterns = {
            "갈취 묻는 말": "",
            "묻고 갈취": "",
        }
        self.korean_recovery_mapping.update(problematic_patterns)

    def detect_critical_repetitive_patterns(self, text: str) -> bool:
        """문제 패턴 감지"""
        if not text or len(text) < 20:
            return False

        critical_patterns = [
            r"갈취 묻는 말",
            r"묻고 갈취",
            r"(.{1,3})\s*(\1\s*){12,}",
        ]

        for pattern in critical_patterns:
            if re.search(pattern, text):
                return True

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

        problematic_removals = [
            "갈취 묻는 말",
            "묻고 갈취",
        ]

        for pattern in problematic_removals:
            text = text.replace(pattern, "")

        words = text.split()
        cleaned_words = []
        i = 0
        while i < len(words):
            current_word = words[i]

            count = 1
            while i + count < len(words) and words[i + count] == current_word:
                count += 1

            if len(current_word) <= 2:
                cleaned_words.extend([current_word] * min(3, count))
            elif len(current_word) <= 5:
                cleaned_words.extend([current_word] * min(5, count))
            elif count >= 10:
                cleaned_words.extend([current_word] * min(5, count))
            else:
                cleaned_words.extend([current_word] * count)

            i += count

        text = " ".join(cleaned_words)

        text = re.sub(r"(.{3,15})\s*\1\s*\1\s*\1\s*\1\s*\1+", r"\1", text)
        text = re.sub(r"(.{1,5})\s*(\1\s*){8,}", r"\1", text)

        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"\s*\(\s*\)\s*", " ", text)

        text = re.sub(r"\s+", " ", text).strip()

        return text

    def restore_korean_characters(self, text: str) -> str:
        """한국어 문자 복구"""
        if not text:
            return ""

        if self.detect_critical_repetitive_patterns(text):
            text = self.remove_critical_repetitive_patterns(text)

        text = unicodedata.normalize("NFC", text)

        for broken, correct in self.korean_recovery_mapping.items():
            text = text.replace(broken, correct)

        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"[.,!?]{3,}", ".", text)
        text = re.sub(r"\s+[.,!?]\s+", ". ", text)

        return text

    def enhance_korean_text_quality(self, text: str) -> str:
        """한국어 텍스트 품질 향상"""
        if not text:
            return ""

        if self.detect_critical_repetitive_patterns(text):
            text = self.remove_critical_repetitive_patterns(text)

        text = self.restore_korean_characters(text)

        for pattern_config in self.korean_quality_patterns:
            pattern = pattern_config["pattern"]
            replacement = pattern_config["replacement"]
            text = re.sub(pattern, replacement, text)

        text = re.sub(r"[^\w\s가-힣.,!?()[\]\-:;/]", " ", text)

        text = re.sub(r"\(\s*\)\s*[가-힣]{1,3}", "", text)

        text = re.sub(r"\s+", " ", text).strip()

        if self.detect_critical_repetitive_patterns(text):
            text = self.remove_critical_repetitive_patterns(text)

        return text

    def fix_grammatical_structure(self, text: str) -> str:
        """문법 구조 수정"""
        if not text:
            return ""

        if self.detect_critical_repetitive_patterns(text):
            text = self.remove_critical_repetitive_patterns(text)

        grammar_fixes = [
            (r"([가-힣])\s+은\s+", r"\1은 "),
            (r"([가-힣])\s+는\s+", r"\1는 "),
            (r"([가-힣])\s+이\s+", r"\1이 "),
            (r"([가-힣])\s+가\s+", r"\1가 "),
            (r"([가-힣])\s+을\s+", r"\1을 "),
            (r"([가-힣])\s+를\s+", r"\1를 "),
            (r"([가-힣])\s+에\s+", r"\1에 "),
            (r"([가-힣])\s+의\s+", r"\1의 "),
            (r"([가-힣])\s+와\s+", r"\1와 "),
            (r"([가-힣])\s+과\s+", r"\1과 "),
            (r"([가-힣])\s+로\s+", r"\1로 "),
            (r"([가-힣])\s+으로\s+", r"\1으로 "),
            (r"([가-힣])\s+다\s*\.", r"\1다."),
            (r"([가-힣])\s+요\s*\.", r"\1요."),
            (r"([가-힣])\s+함\s*\.", r"\1함."),
            (r"([가-힣])\s+니다\s*\.", r"\1니다."),
            (r"([가-힣])\s+습니다\s*\.", r"\1습니다."),
            (r"\.+", "."),
            (r"\s*\.\s*", ". "),
            (r"\s*,\s*", ", "),
            (r"([가-힣])\s*$", r"\1."),
        ]

        for pattern, replacement in grammar_fixes:
            if callable(replacement):
                text = re.sub(pattern, replacement, text)
            else:
                text = re.sub(pattern, replacement, text)

        sentences = text.split(".")
        improved_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            if self.detect_critical_repetitive_patterns(sentence):
                continue

            sentence = sentence.lstrip()

            if len(sentence) > 250:
                parts = re.split(r"[,，]", sentence)
                if len(parts) > 1:
                    for part in parts:
                        part = part.strip()
                        if len(
                            part
                        ) > 8 and not self.detect_critical_repetitive_patterns(part):
                            improved_sentences.append(part)
                else:
                    if not self.detect_critical_repetitive_patterns(sentence):
                        improved_sentences.append(sentence)
            else:
                if not self.detect_critical_repetitive_patterns(sentence):
                    improved_sentences.append(sentence)

        if improved_sentences:
            result = ". ".join(improved_sentences)
        else:
            result = "관련 법령과 규정에 따라 체계적인 관리를 수행해야 합니다"

        if result and not result.endswith("."):
            result += "."

        return result

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

        intent_scores = {}

        for intent_type, patterns in self.question_intent_patterns.items():
            score = 0
            matched_patterns = []

            for pattern in patterns:
                matches = re.findall(pattern, question, re.IGNORECASE)
                if matches:
                    if len(matches) > 1:
                        score += 2.0
                    else:
                        score += 1.5
                    matched_patterns.append(pattern)

            keyword_bonuses = {
                "기관_묻기": ["기관", "위원회", "담당", "업무", "어디", "누가"],
                "특징_묻기": ["특징", "특성", "성질", "속성", "어떤"],
                "지표_묻기": ["지표", "징후", "탐지", "모니터링", "신호"],
                "방안_묻기": ["방안", "대책", "해결", "대응", "어떻게"],
                "절차_묻기": ["절차", "과정", "단계", "순서", "프로세스"],
                "조치_묻기": ["조치", "대응", "예방", "보안"],
            }

            if intent_type in keyword_bonuses:
                keyword_matches = sum(
                    1
                    for keyword in keyword_bonuses[intent_type]
                    if keyword in question_lower
                )
                if keyword_matches > 0:
                    score += keyword_matches * 0.5

            if score > 0:
                intent_scores[intent_type] = {
                    "score": score,
                    "patterns": matched_patterns,
                }

        if intent_scores:
            sorted_intents = sorted(
                intent_scores.items(), key=lambda x: x[1]["score"], reverse=True
            )
            best_intent = sorted_intents[0]

            intent_analysis["primary_intent"] = best_intent[0]
            intent_analysis["intent_confidence"] = min(
                best_intent[1]["score"] / 3.0, 1.0
            )
            intent_analysis["detected_patterns"] = best_intent[1]["patterns"]

            if len(sorted_intents) > 1:
                intent_analysis["secondary_intents"] = [
                    {"intent": intent, "score": data["score"]}
                    for intent, data in sorted_intents[1:3]
                ]

            primary = best_intent[0]
            if "기관" in primary:
                intent_analysis["answer_type_required"] = "기관명"
                intent_analysis["context_hints"].append("구체적인 기관명 필요")
            elif "특징" in primary:
                intent_analysis["answer_type_required"] = "특징설명"
                intent_analysis["context_hints"].append("특징과 성질 나열")
            elif "지표" in primary:
                intent_analysis["answer_type_required"] = "지표나열"
                intent_analysis["context_hints"].append("탐지 지표와 징후")
            elif "방안" in primary:
                intent_analysis["answer_type_required"] = "방안제시"
                intent_analysis["context_hints"].append("구체적 실행방안")
            elif "절차" in primary:
                intent_analysis["answer_type_required"] = "절차설명"
                intent_analysis["context_hints"].append("단계별 절차")
            elif "조치" in primary:
                intent_analysis["answer_type_required"] = "조치설명"
                intent_analysis["context_hints"].append("보안조치 내용")
            elif "법령" in primary:
                intent_analysis["answer_type_required"] = "법령설명"
                intent_analysis["context_hints"].append("관련 법령과 규정")
            elif "정의" in primary:
                intent_analysis["answer_type_required"] = "정의설명"
                intent_analysis["context_hints"].append("개념과 정의")

        self._add_context_analysis(question, intent_analysis)

        return intent_analysis

    def _add_context_analysis(self, question: str, intent_analysis: Dict):
        """문맥 분석 추가"""
        question_lower = question.lower()

        urgency_keywords = ["긴급", "즉시", "신속", "빠른"]
        if any(keyword in question_lower for keyword in urgency_keywords):
            intent_analysis["context_hints"].append("긴급 대응 필요")

        example_keywords = ["예시", "사례", "구체적", "실제"]
        if any(keyword in question_lower for keyword in example_keywords):
            intent_analysis["context_hints"].append("구체적 예시 포함")

        comparison_keywords = ["비교", "차이", "구별", "비교하여"]
        if any(keyword in question_lower for keyword in comparison_keywords):
            intent_analysis["context_hints"].append("비교 분석 필요")

        step_keywords = ["단계", "순서", "과정", "절차"]
        if any(keyword in question_lower for keyword in step_keywords):
            intent_analysis["context_hints"].append("단계별 설명 필요")

    def extract_choice_range(self, question: str) -> Tuple[str, int]:
        """선택지 범위 추출"""
        question_type = self.analyze_question_type(question)

        if question_type != "multiple_choice":
            return "subjective", 0

        lines = question.split("\n")
        choice_numbers = []

        for line in lines:
            line = line.strip()
            match = re.match(r"^(\d+)\s+(.+)", line)
            if match:
                num = int(match.group(1))
                content = match.group(2).strip()
                if 1 <= num <= 5 and len(content) > 0:
                    choice_numbers.append(num)

        if choice_numbers:
            choice_numbers.sort()
            max_choice = max(choice_numbers)
            min_choice = min(choice_numbers)

            expected_count = max_choice - min_choice + 1
            if (
                len(choice_numbers) == expected_count
                and min_choice == 1
                and max_choice >= 3
            ):
                return "multiple_choice", max_choice

        for i in range(5, 2, -1):
            pattern_parts = [f"{j}\\s+[가-힣\\w]+" for j in range(1, i + 1)]
            pattern = ".*".join(pattern_parts)
            if re.search(pattern, question, re.DOTALL):
                return "multiple_choice", i

        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                return "multiple_choice", 5

        return "subjective", 0

    def analyze_question_type(self, question: str) -> str:
        """질문 유형 분석"""

        question = question.strip()

        for pattern in self.subj_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return "subjective"

        choice_pattern = r"\n(\d+)\s+[가-힣\w]"
        choice_matches = re.findall(choice_pattern, question)

        if len(choice_matches) >= 3:
            choice_nums = [int(match) for match in choice_matches]
            choice_nums.sort()
            if (
                choice_nums[0] == 1
                and len(choice_nums) == choice_nums[-1]
                and choice_nums[-1] <= 5
            ):
                return "multiple_choice"

        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                if any(f"{i} " in question for i in range(1, 6)):
                    return "multiple_choice"

        for pattern in self.mc_patterns:
            if re.search(pattern, question, re.DOTALL | re.MULTILINE):
                return "multiple_choice"

        if (
            len(question) < 400
            and re.search(r"것은\?|것\?|것은\s*$", question)
            and len(re.findall(r"\b[1-5]\b", question)) >= 3
        ):
            return "multiple_choice"

        return "subjective"

    def extract_domain(self, question: str) -> str:
        """도메인 추출"""
        question_lower = question.lower()

        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    if keyword in [
                        "개인정보보호법",
                        "전자금융거래법",
                        "자본시장법",
                        "ISMS",
                        "트로이",
                        "RAT",
                        "원격제어",
                        "분쟁조정",
                        "위험관리",
                    ]:
                        score += 5
                    elif keyword in [
                        "개인정보",
                        "전자금융",
                        "금융투자",
                        "사이버보안",
                        "정보보안",
                        "위험관리",
                    ]:
                        score += 3
                    else:
                        score += 1

            if score > 0:
                domain_scores[domain] = score

        if not domain_scores:
            return "일반"

        detected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]

        if detected_domain == "사이버보안":
            cybersec_keywords = [
                "트로이",
                "악성코드",
                "RAT",
                "원격제어",
                "딥페이크",
                "SBOM",
                "보안",
            ]
            if any(keyword in question_lower for keyword in cybersec_keywords):
                detected_domain = "사이버보안"
        elif detected_domain == "개인정보보호":
            privacy_keywords = ["개인정보", "정보주체", "만 14세", "법정대리인", "PIMS"]
            if any(keyword in question_lower for keyword in privacy_keywords):
                detected_domain = "개인정보보호"

        return detected_domain

    def clean_korean_text(self, text: str) -> str:
        """한국어 텍스트 정리"""
        if not text:
            return ""

        if self.detect_critical_repetitive_patterns(text):
            text = self.remove_critical_repetitive_patterns(text)
            if len(text) < 8:
                return "텍스트 정리 중 내용이 부족합니다."

        text = self.restore_korean_characters(text)

        text = self.enhance_korean_text_quality(text)

        text = self.fix_grammatical_structure(text)

        text = re.sub(r"\s+", " ", text).strip()

        text = re.sub(r"[^\w\s가-힣.,!?()[\]\-]", " ", text)

        english_chars = len(re.findall(r"[a-zA-Z]", text))
        total_chars = len(re.sub(r"[^\w가-힣]", "", text))
        if total_chars > 0 and english_chars / total_chars > 0.4:
            text = re.sub(r"[a-zA-Z]+", "", text)

        text = re.sub(r"[\u4e00-\u9fff]", "", text)

        text = re.sub(r"[①②③④⑤➀➁➂➃➄]", "", text)

        text = re.sub(r"\s+", " ", text).strip()

        if self.detect_critical_repetitive_patterns(text):
            text = self.remove_critical_repetitive_patterns(text)
            if len(text) < 10:
                return "텍스트 정리 후 내용이 부족합니다."

        return text

    def calculate_korean_ratio(self, text: str) -> float:
        """한국어 비율 계산"""
        if not text:
            return 0.0

        korean_chars = len(re.findall(r"[가-힣]", text))
        total_chars = len(re.sub(r"[^\w가-힣]", "", text))

        if total_chars == 0:
            return 0.0

        return korean_chars / total_chars

    def calculate_english_ratio(self, text: str) -> float:
        """영어 비율 계산"""
        if not text:
            return 0.0

        english_chars = len(re.findall(r"[a-zA-Z]", text))
        total_chars = len(re.sub(r"[^\w가-힣]", "", text))

        if total_chars == 0:
            return 0.0

        return english_chars / total_chars

    def validate_mc_answer_range(self, answer: str, max_choice: int) -> bool:
        """객관식 답변 범위 확인"""
        if not answer or not answer.isdigit():
            return False

        answer_num = int(answer)
        return 1 <= answer_num <= max_choice

    def validate_answer_intent_match(
        self, answer: str, question: str, intent_analysis: Dict
    ) -> bool:
        """답변과 의도 매칭 검증"""
        if not answer or not intent_analysis:
            return False

        if self.detect_critical_repetitive_patterns(answer):
            return False

        required_type = intent_analysis.get("answer_type_required", "설명형")
        answer_lower = answer.lower()

        if required_type == "기관명":
            institution_keywords = [
                "위원회",
                "감독원",
                "은행",
                "기관",
                "센터",
                "청",
                "부",
                "원",
                "전자금융분쟁조정위원회",
                "금융감독원",
                "개인정보보호위원회",
                "한국은행",
                "금융위원회",
                "과학기술정보통신부",
                "개인정보침해신고센터",
                "담당",
                "업무",
                "수행",
            ]

            specific_institutions = [
                "전자금융분쟁조정위원회",
                "금융감독원",
                "개인정보보호위원회",
                "개인정보침해신고센터",
                "한국은행",
                "금융위원회",
            ]

            has_specific = any(inst in answer_lower for inst in specific_institutions)
            keyword_count = sum(
                1 for keyword in institution_keywords if keyword in answer_lower
            )

            match_found = has_specific or keyword_count >= 1

        elif required_type == "특징설명":
            feature_keywords = [
                "특징",
                "특성",
                "속성",
                "성질",
                "기능",
                "역할",
                "원리",
                "성격",
                "특색",
                "성질",
                "특점",
                "양상",
            ]
            descriptive_words = [
                "위장",
                "은밀",
                "지속",
                "제어",
                "접근",
                "수행",
                "활동",
                "작동",
                "동작",
                "기능",
                "역할",
                "처리",
                "관리",
            ]

            feature_count = sum(
                1 for keyword in feature_keywords if keyword in answer_lower
            )
            desc_count = sum(1 for word in descriptive_words if word in answer_lower)

            match_found = (
                feature_count >= 1 or desc_count >= 1
            ) and not self.detect_critical_repetitive_patterns(answer)

        elif required_type == "지표나열":
            indicator_keywords = [
                "지표",
                "신호",
                "징후",
                "패턴",
                "행동",
                "활동",
                "모니터링",
                "탐지",
                "발견",
                "식별",
                "관찰",
                "분석",
                "추적",
                "감시",
            ]
            specific_indicators = [
                "네트워크",
                "트래픽",
                "프로세스",
                "파일",
                "시스템",
                "로그",
                "연결",
                "접근",
                "사용",
                "변경",
                "수정",
                "생성",
            ]

            indicator_count = sum(
                1 for keyword in indicator_keywords if keyword in answer_lower
            )
            specific_count = sum(
                1 for word in specific_indicators if word in answer_lower
            )

            match_found = (
                indicator_count >= 1 or specific_count >= 1
            ) and not self.detect_critical_repetitive_patterns(answer)

        elif required_type == "방안제시":
            solution_keywords = [
                "방안",
                "대책",
                "조치",
                "해결",
                "대응",
                "관리",
                "처리",
                "절차",
                "개선",
                "예방",
                "보완",
                "강화",
                "구축",
                "수립",
                "마련",
            ]
            action_words = [
                "수립",
                "구축",
                "시행",
                "실시",
                "강화",
                "개선",
                "마련",
                "준비",
                "실행",
                "진행",
                "처리",
                "관리",
                "운영",
            ]

            solution_count = sum(
                1 for keyword in solution_keywords if keyword in answer_lower
            )
            action_count = sum(1 for word in action_words if word in answer_lower)

            match_found = (
                solution_count >= 1 or action_count >= 1
            ) and not self.detect_critical_repetitive_patterns(answer)

        elif required_type == "절차설명":
            procedure_keywords = [
                "절차",
                "과정",
                "단계",
                "순서",
                "프로세스",
                "진행",
                "수행",
                "실행",
                "처리",
                "진행",
                "실시",
                "수립",
            ]
            step_indicators = [
                "첫째",
                "둘째",
                "먼저",
                "다음",
                "마지막",
                "단계적",
                "순차적",
                "차례",
                "순서",
                "과정",
                "절차",
            ]

            proc_count = sum(
                1 for keyword in procedure_keywords if keyword in answer_lower
            )
            step_count = sum(1 for word in step_indicators if word in answer_lower)

            match_found = (
                proc_count >= 1 or step_count >= 1 or "," in answer
            ) and not self.detect_critical_repetitive_patterns(answer)

        elif required_type == "조치설명":
            measure_keywords = [
                "조치",
                "대응",
                "대책",
                "방안",
                "보안",
                "예방",
                "개선",
                "강화",
                "보완",
                "관리",
                "처리",
                "수행",
                "시행",
                "실시",
            ]
            match_found = sum(
                1 for keyword in measure_keywords if keyword in answer_lower
            ) >= 1 and not self.detect_critical_repetitive_patterns(answer)

        elif required_type == "법령설명":
            law_keywords = [
                "법",
                "법령",
                "법률",
                "규정",
                "조항",
                "규칙",
                "기준",
                "근거",
                "조례",
                "시행령",
                "고시",
                "지침",
            ]
            match_found = sum(
                1 for keyword in law_keywords if keyword in answer_lower
            ) >= 1 and not self.detect_critical_repetitive_patterns(answer)

        elif required_type == "정의설명":
            definition_keywords = [
                "정의",
                "개념",
                "의미",
                "뜻",
                "용어",
                "정의",
                "설명",
                "해석",
            ]
            match_found = sum(
                1 for keyword in definition_keywords if keyword in answer_lower
            ) >= 1 and not self.detect_critical_repetitive_patterns(answer)

        else:
            meaningful_words = [
                "법령",
                "규정",
                "관리",
                "조치",
                "절차",
                "기준",
                "정책",
                "체계",
                "시스템",
                "업무",
                "담당",
                "수행",
                "필요",
                "해야",
                "구축",
                "수립",
                "시행",
                "실시",
                "강화",
                "개선",
                "확보",
                "보장",
            ]
            match_found = sum(
                1 for word in meaningful_words if word in answer_lower
            ) >= 1 and not self.detect_critical_repetitive_patterns(answer)

        return match_found

    def validate_korean_answer(
        self, answer: str, question_type: str, max_choice: int = 5, question: str = ""
    ) -> bool:
        """한국어 답변 검증"""
        if not answer:
            return False

        answer = str(answer).strip()

        if self.detect_critical_repetitive_patterns(answer):
            return False

        if question_type == "multiple_choice":
            if not self.validate_mc_answer_range(answer, max_choice):
                return False
            return True

        else:
            clean_answer = self.clean_korean_text(answer)

            if self.detect_critical_repetitive_patterns(clean_answer):
                return False

            if not (self.korean_requirements["min_length"] <= len(clean_answer) <= 700):
                return False

            korean_ratio = self.calculate_korean_ratio(clean_answer)
            if korean_ratio < self.korean_requirements["min_korean_ratio"]:
                return False

            english_ratio = self.calculate_english_ratio(answer)
            if english_ratio > self.korean_requirements["max_english_ratio"]:
                return False

            korean_chars = len(re.findall(r"[가-힣]", clean_answer))
            if korean_chars < 10:
                return False

            meaningful_keywords = [
                "법",
                "규정",
                "조치",
                "관리",
                "보안",
                "방안",
                "절차",
                "기준",
                "정책",
                "체계",
                "시스템",
                "통제",
                "특징",
                "지표",
                "탐지",
                "대응",
                "기관",
                "위원회",
                "감독원",
                "업무",
                "담당",
                "수행",
                "필요",
                "해야",
                "구축",
                "수립",
                "시행",
                "실시",
                "강화",
                "개선",
                "확보",
                "보장",
            ]
            if not any(word in clean_answer for word in meaningful_keywords):
                return False

            if question:
                intent_analysis = self.analyze_question_intent(question)
                if not self.validate_answer_intent_match(
                    answer, question, intent_analysis
                ):
                    if len(clean_answer) >= 25 and korean_ratio >= 0.6:
                        return True
                    return False

            return True

    def validate_answer(
        self, answer: str, question_type: str, max_choice: int = 5, question: str = ""
    ) -> bool:
        """답변 검증"""
        return self.validate_korean_answer(answer, question_type, max_choice, question)

    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        return self.clean_korean_text(text)

    def extract_choices(self, question: str) -> List[str]:
        """선택지 추출"""
        choices = []

        lines = question.split("\n")
        for line in lines:
            line = line.strip()
            match = re.match(r"^(\d+)\s+(.+)", line)
            if match:
                choice_num = int(match.group(1))
                choice_content = match.group(2).strip()
                if 1 <= choice_num <= 5 and len(choice_content) > 0:
                    choices.append(choice_content)

        if len(choices) >= 3:
            return choices

        if not choices:
            patterns = [
                r"(\d+)\s+([^0-9\n]+?)(?=\d+\s+|$)",
                r"(\d+)\)\s*([^0-9\n]+?)(?=\d+\)|$)",
                r"(\d+)\.\s*([^0-9\n]+?)(?=\d+\.|$)",
                r"[①②③④⑤]\s*([^①②③④⑤\n]+?)(?=[①②③④⑤]|$)",
            ]

            for pattern in patterns:
                matches = re.findall(pattern, question, re.MULTILINE | re.DOTALL)
                if matches:
                    if isinstance(matches[0], tuple):
                        choices = [match[1].strip() for match in matches]
                    else:
                        choices = [match.strip() for match in matches]

                    if len(choices) >= 3:
                        break

        return choices[:5]

    def analyze_question_difficulty(self, question: str) -> str:
        """질문 난이도 분석"""
        question_lower = question.lower()

        technical_terms = [
            "isms",
            "pims",
            "sbom",
            "원격제어",
            "침입탐지",
            "트로이",
            "멀웨어",
            "랜섬웨어",
            "딥페이크",
            "피싱",
            "접근매체",
            "전자서명",
            "개인정보보호법",
            "자본시장법",
            "rat",
            "원격접근",
            "탐지지표",
            "apt",
            "ddos",
            "ids",
            "ips",
            "bcp",
            "drp",
            "isms-p",
            "분쟁조정",
            "금융투자업",
            "위험관리",
            "재해복구",
            "비상연락체계",
        ]

        term_count = sum(1 for term in technical_terms if term in question_lower)

        length = len(question)

        choice_count = len(self.extract_choices(question))

        if term_count >= 2 or length > 350 or choice_count >= 5:
            return "고급"
        elif term_count >= 1 or length > 150 or choice_count >= 4:
            return "중급"
        else:
            return "초급"

    def normalize_korean_answer(
        self, answer: str, question_type: str, max_choice: int = 5
    ) -> str:
        """한국어 답변 정규화"""
        if not answer:
            return ""

        answer = str(answer).strip()

        if question_type == "multiple_choice":
            numbers = re.findall(r"[1-9]", answer)
            for num in numbers:
                if 1 <= int(num) <= max_choice:
                    return num

            return ""

        else:
            answer = self.clean_korean_text(answer)

            if self.detect_critical_repetitive_patterns(answer):
                return "답변 생성 중 반복 패턴이 감지되어 재생성이 필요합니다."

            if len(answer) < 10:
                return "답변 길이가 부족하여 생성에 실패했습니다."

            if len(answer) > 700:
                sentences = answer.split(". ")
                valid_sentences = []

                for sentence in sentences:
                    if not self.detect_critical_repetitive_patterns(sentence):
                        valid_sentences.append(sentence)
                    if len(valid_sentences) >= 5:
                        break

                if valid_sentences:
                    answer = ". ".join(valid_sentences[:5])
                else:
                    return "답변 정규화 중 유효한 문장을 찾을 수 없습니다."

                if len(answer) > 700:
                    answer = answer[:700]

            if answer and not answer.endswith((".", "다", "요", "함")):
                answer += "."

            return answer

    def normalize_answer(
        self, answer: str, question_type: str, max_choice: int = 5
    ) -> str:
        """답변 정규화"""
        return self.normalize_korean_answer(answer, question_type, max_choice)

    def cleanup(self):
        """리소스 정리"""
        pass