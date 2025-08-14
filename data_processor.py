# data_processor.py

"""
실제 데이터 처리 시스템 - 수정됨
- 질문 구조 분석 개선
- 객관식/주관식 분류 정확도 향상
- 텍스트 전처리 및 정리
- 도메인 분류
- 한국어 품질 관리
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd

class RealDataProcessor:
    """실제 데이터 처리기 - 대회 규칙 준수, 분류 정확도 개선"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.domain_keywords = {
            "개인정보": ["개인정보", "정보주체", "개인정보보호법", "민감정보", "고유식별정보"],
            "전자금융": ["전자금융", "전자적", "접근매체", "전자금융거래법", "전자서명"],
            "사이버보안": ["트로이", "악성코드", "해킹", "멀웨어", "피싱", "스미싱", "랜섬웨어"],
            "정보보안": ["정보보안", "보안관리", "ISMS", "보안정책", "접근통제", "암호화"]
        }
        
        # 객관식 패턴 강화
        self.mc_question_patterns = [
            r'해당하지.*않는.*것',
            r'적절하지.*않는.*것',
            r'옳지.*않는.*것',
            r'틀린.*것',
            r'맞는.*것',
            r'옳은.*것',
            r'적절한.*것',
            r'해당하는.*것',
            r'올바른.*것',
            r'바르지.*않는.*것',
            r'설명.*중.*옳은.*것',
            r'설명.*중.*틀린.*것',
            r'다음.*중.*해당',
            r'다음.*중.*적절',
            r'다음.*중.*옳',
            r'다음.*중.*맞',
            r'다음.*중.*틀린',
            r'다음.*중.*바른',
            r'보기.*중.*옳',
            r'보기.*중.*적절',
            r'설명.*옳은.*것',
            r'설명.*틀린.*것'
        ]
        
    def analyze_question_structure(self, question: str) -> Dict:
        """질문 구조 분석 - 개선된 객관식/주관식 분류"""
        question = question.strip()
        
        # 1차: 선택지 기호 패턴 검사 (가장 확실한 방법)
        choice_patterns = [
            r'①.*②.*③.*④.*⑤',
            r'1\).*2\).*3\).*4\).*5\)',
            r'가\).*나\).*다\).*라\).*마\)',
            r'ㄱ\).*ㄴ\).*ㄷ\).*ㄹ\).*ㅁ\)'
        ]
        
        has_choice_symbols = any(re.search(pattern, question, re.DOTALL | re.IGNORECASE) for pattern in choice_patterns)
        
        # 2차: 객관식 질문 패턴 검사 (강화됨)
        has_mc_pattern = any(re.search(pattern, question, re.IGNORECASE) for pattern in self.mc_question_patterns)
        
        # 3차: 질문 종료 패턴 검사
        ending_patterns = [
            r'것은\?',
            r'것은\s*$',
            r'것\?',
            r'것\s*$'
        ]
        has_mc_ending = any(re.search(pattern, question, re.IGNORECASE) for pattern in ending_patterns)
        
        # 4차: 부정형 질문 패턴 (객관식 가능성 높음)
        negative_patterns = [
            r'아닌.*것',
            r'않는.*것',
            r'없는.*것',
            r'못한.*것',
            r'잘못.*것'
        ]
        has_negative = any(re.search(pattern, question, re.IGNORECASE) for pattern in negative_patterns)
        
        # 종합 판단: 선택지 기호가 있으면 무조건 객관식
        if has_choice_symbols:
            is_multiple_choice = True
        # 그 외에는 패턴 점수로 판단
        else:
            mc_score = 0
            if has_mc_pattern:
                mc_score += 3
            if has_mc_ending:
                mc_score += 2
            if has_negative:
                mc_score += 2
            
            # 금융보안 대회의 경우 대부분 객관식이므로 임계값 낮춤
            is_multiple_choice = mc_score >= 2
        
        # 복잡도 점수 계산
        complexity_score = self._calculate_complexity(question)
        
        # 선택지 분석
        choices = []
        if is_multiple_choice:
            choices = self._extract_choices_enhanced(question)
        
        return {
            "question_type": "multiple_choice" if is_multiple_choice else "subjective",
            "complexity_score": complexity_score,
            "length": len(question),
            "has_legal_terms": self._has_legal_terms(question),
            "domain_hints": self._extract_domain_hints(question),
            "has_negative": has_negative,
            "technical_level": self._assess_technical_level(question),
            "choices": choices,
            "choice_count": len(choices),
            "choice_analysis": self._analyze_choices(choices) if choices else {},
            "mc_confidence": mc_score if not has_choice_symbols else 5  # 신뢰도 점수
        }
    
    def _extract_choices_enhanced(self, question: str) -> List[Dict]:
        """향상된 선택지 추출"""
        choices = []
        
        # 기호별 패턴 (순서대로)
        symbol_patterns = [
            (r'①\s*([^②③④⑤]+)', "1"),
            (r'②\s*([^①③④⑤]+)', "2"), 
            (r'③\s*([^①②④⑤]+)', "3"),
            (r'④\s*([^①②③⑤]+)', "4"),
            (r'⑤\s*([^①②③④]+)', "5"),
            (r'1\)\s*([^2)3)4)5)]+)', "1"),
            (r'2\)\s*([^1)3)4)5)]+)', "2"),
            (r'3\)\s*([^1)2)4)5)]+)', "3"),
            (r'4\)\s*([^1)2)3)5)]+)', "4"),
            (r'5\)\s*([^1)2)3)4)]+)', "5"),
            (r'가\)\s*([^나)다)라)마)]+)', "1"),
            (r'나\)\s*([^가)다)라)마)]+)', "2"),
            (r'다\)\s*([^가)나)라)마)]+)', "3"),
            (r'라\)\s*([^가)나)다)마)]+)', "4"),
            (r'마\)\s*([^가)나)다)라)]+)', "5")
        ]
        
        for pattern, number in symbol_patterns:
            match = re.search(pattern, question, re.DOTALL)
            if match:
                choice_text = match.group(1).strip()
                if choice_text and len(choice_text) > 2:  # 최소 길이 확인
                    choices.append({
                        "number": number,
                        "text": choice_text[:100],  # 최대 100자
                        "length": len(choice_text)
                    })
        
        # 선택지 개수가 5개가 아니면 빈 리스트 반환 (신뢰도 낮음)
        if len(choices) != 5:
            if self.debug_mode:
                print(f"선택지 개수 부족: {len(choices)}개")
            return []
        
        return choices
    
    def _analyze_choices(self, choices: List[Dict]) -> Dict:
        """선택지 분석"""
        if not choices:
            return {}
        
        analysis = {
            "total_choices": len(choices),
            "avg_length": np.mean([c["length"] for c in choices]),
            "balanced": True,  # 선택지 길이 균형
            "inclusion_candidates": [],  # 포함 관계 후보
            "exclusion_candidates": []   # 배제 관계 후보
        }
        
        # 길이 균형 체크
        lengths = [c["length"] for c in choices]
        if np.std(lengths) > np.mean(lengths) * 0.5:
            analysis["balanced"] = False
        
        # 부정형 질문에서 배제 후보 찾기
        for choice in choices:
            choice_text = choice["text"].lower()
            if any(word in choice_text for word in ["모두", "전부", "모든", "모든것"]):
                analysis["exclusion_candidates"].append(choice["number"])
            elif any(word in choice_text for word in ["없다", "아니다", "해당없음"]):
                analysis["exclusion_candidates"].append(choice["number"])
        
        return analysis
    
    def _calculate_complexity(self, question: str) -> float:
        """질문 복잡도 계산 - 개선됨"""
        complexity_score = 0
        
        # 기본 길이 기반 (가중치 조정)
        complexity_score += min(len(question) * 0.0008, 0.3)
        
        # 전문 용어 개수
        professional_terms = len(re.findall(r'[가-힣]{4,}', question))
        complexity_score += min(professional_terms * 0.02, 0.2)
        
        # 법률 용어 (가중치 증가)
        legal_terms = ['법', '규정', '조치', '의무', '책임', '처벌', '위반', '준수', '시행']
        legal_count = sum(question.count(term) for term in legal_terms)
        complexity_score += min(legal_count * 0.08, 0.25)
        
        # 기술 용어
        tech_terms = ['시스템', '프로그램', '네트워크', '데이터베이스', '서버', '암호화', '인증']
        tech_count = sum(question.count(term) for term in tech_terms)
        complexity_score += min(tech_count * 0.05, 0.15)
        
        # 문장 구조 복잡도
        sentence_count = question.count('.') + question.count('?') + question.count('!') + 1
        if sentence_count > 2:
            complexity_score += min((sentence_count - 2) * 0.05, 0.1)
        
        # 선택지 존재 시 복잡도 증가
        if any(symbol in question for symbol in ['①', '②', '③', '④', '⑤']):
            complexity_score += 0.1
        
        return min(complexity_score, 1.0)
    
    def _has_legal_terms(self, question: str) -> bool:
        """법률 용어 포함 여부 - 확장됨"""
        legal_terms = [
            '법', '규정', '조치', '의무', '책임', '처벌', '위반', '준수', '시행', '적용',
            '조항', '규칙', '지침', '기준', '표준', '절차', '정책', '제도'
        ]
        return any(term in question for term in legal_terms)
    
    def _extract_domain_hints(self, question: str) -> List[str]:
        """도메인 힌트 추출 - 향상됨"""
        hints = []
        question_lower = question.lower()
        
        # 우선순위별로 체크 (더 구체적인 것부터)
        domain_priority = [
            ("사이버보안", ["트로이", "악성코드", "해킹", "멀웨어", "피싱", "스미싱", "랜섬웨어", "바이러스", "웜", "스파이웨어"]),
            ("개인정보", ["개인정보", "정보주체", "개인정보보호법", "민감정보", "고유식별정보", "수집", "이용", "제공", "파기"]),
            ("전자금융", ["전자금융", "전자적", "접근매체", "전자금융거래법", "전자서명", "전자인증", "공인인증서"]),
            ("정보보안", ["정보보안", "보안관리", "ISMS", "보안정책", "접근통제", "암호화", "방화벽", "침입탐지"])
        ]
        
        for domain, keywords in domain_priority:
            if any(keyword in question_lower for keyword in keywords):
                hints.append(domain)
                break  # 첫 번째 매칭된 도메인만 사용
        
        return hints if hints else ["일반"]
    
    def _assess_technical_level(self, question: str) -> str:
        """기술적 난이도 평가 - 세분화됨"""
        technical_keywords = {
            "고급": ["암호화알고리즘", "해시함수", "디지털서명", "PKI", "SSL/TLS", "VPN", "침입탐지시스템", "방화벽정책"],
            "중급": ["방화벽", "IDS", "VPN", "접근제어", "인증", "권한관리", "로그분석", "취약점"],
            "초급": ["비밀번호", "백업", "업데이트", "바이러스", "보안", "사용자교육", "정책수립"]
        }
        
        question_lower = question.lower()
        
        for level, keywords in technical_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return level
        
        # 복잡도 기반 판단
        complexity = self._calculate_complexity(question)
        if complexity > 0.7:
            return "고급"
        elif complexity > 0.4:
            return "중급"
        else:
            return "초급"
    
    def _clean_korean_text(self, text: str) -> str:
        """한국어 텍스트 정리 - 강화됨"""
        if not text:
            return ""
        
        # 기본 정리
        text = re.sub(r'[①②③④⑤➀➁➂➃➄]', '', text)
        text = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]+(?![가-힣])', '', text)  # 단독 자음/모음만 제거
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 영어 단어 정리 (전문용어 제외) - 확장됨
        preserve_words = [
            'IT', 'AI', 'API', 'DB', 'OS', 'IP', 'DNS', 'HTTP', 'HTTPS', 'SSL', 'TLS', 
            'VPN', 'PKI', 'ISMS', 'ISO', 'NIST', 'OWASP', 'CSRF', 'XSS', 'SQL', 'DDoS',
            'IDS', 'IPS', 'SIEM', 'MDM', 'DLP', 'NAC', 'UTM', 'WAF', 'CERT'
        ]
        
        # 보존할 단어들을 임시로 치환
        temp_replacements = {}
        for i, word in enumerate(preserve_words):
            if word in text:
                temp_key = f"__PRESERVE_{i}__"
                temp_replacements[temp_key] = word
                text = text.replace(word, temp_key)
        
        # 긴 영어 단어 제거 (4글자 이상)
        text = re.sub(r'\b[A-Za-z]{4,}\b', '', text)
        
        # 보존된 단어들 복원
        for temp_key, original_word in temp_replacements.items():
            text = text.replace(temp_key, original_word)
        
        # 특수문자 정리
        text = re.sub(r'[^\w\s가-힣.,!?()-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 최소 길이 확인
        if len(text) < 10:
            return ""
        
        return text
    
    def extract_multiple_choice_options(self, question: str) -> List[str]:
        """객관식 선택지 추출 - 향상됨"""
        choices = self._extract_choices_enhanced(question)
        return [choice["text"] for choice in choices]
    
    def validate_answer_format(self, answer: str, question_type: str) -> Tuple[bool, str]:
        """답변 형식 검증 - 강화됨"""
        if question_type == "multiple_choice":
            # 객관식: 1-5 숫자여야 함
            clean_answer = answer.strip()
            if re.match(r'^[1-5]$', clean_answer):
                return True, clean_answer
            
            # 숫자 추출 시도 (더 정확하게)
            numbers = re.findall(r'[1-5]', answer)
            if numbers:
                return True, numbers[0]
            
            # 한글 번호 변환
            korean_numbers = {'일': '1', '이': '2', '삼': '3', '사': '4', '오': '5'}
            for korean, num in korean_numbers.items():
                if korean in answer:
                    return True, num
            
            return False, ""
        
        else:  # subjective
            # 주관식: 한국어 포함, 최소 길이
            cleaned = self._clean_korean_text(answer)
            
            if len(cleaned) < 15:  # 최소 길이 완화
                return False, ""
            
            # 한국어 비율 확인 (기준 완화)
            korean_chars = len(re.findall(r'[가-힣]', cleaned))
            total_chars = len(re.sub(r'[^\w]', '', cleaned))
            
            if total_chars == 0 or korean_chars / total_chars < 0.5:  # 50%로 완화
                return False, ""
            
            return True, cleaned
    
    def extract_keywords(self, text: str) -> List[str]:
        """키워드 추출 - 확장됨"""
        professional_terms = []
        
        # 법률 용어 (확장됨)
        legal_pattern = r'(개인정보보호법|전자금융거래법|정보통신망법|신용정보법|[가-힣]{2,}법령?|[가-힣]{2,}규정|[가-힣]{2,}지침)'
        legal_terms = re.findall(legal_pattern, text)
        professional_terms.extend(legal_terms)
        
        # 기술 용어 (확장됨)
        tech_pattern = r'(시스템|프로그램|네트워크|데이터베이스|서버|보안|암호화|인증|접근제어|방화벽|침입탐지|취약점|위협|위험|모니터링)'
        tech_terms = re.findall(tech_pattern, text)
        professional_terms.extend(tech_terms)
        
        # 관리 및 정책 용어
        mgmt_pattern = r'([가-힣]{2,}관리|[가-힣]{2,}정책|[가-힣]{2,}절차|[가-힣]{2,}체계|[가-힣]{2,}기준)'
        mgmt_terms = re.findall(mgmt_pattern, text)
        professional_terms.extend(mgmt_terms)
        
        # 보안 관련 동작
        action_pattern = r'([가-힣]{2,}조치|[가-힣]{2,}대응|[가-힣]{2,}점검|[가-힣]{2,}감시|[가-힣]{2,}탐지)'
        action_terms = re.findall(action_pattern, text)
        professional_terms.extend(action_terms)
        
        return list(set(professional_terms))
    
    def classify_question_difficulty(self, question: str) -> str:
        """질문 난이도 분류 - 세분화됨"""
        structure = self.analyze_question_structure(question)
        complexity = structure["complexity_score"]
        technical_level = structure["technical_level"]
        has_choices = len(structure.get("choices", [])) > 0
        
        # 선택지가 있는 객관식은 기본적으로 중급 이상
        base_level = 0.3 if has_choices else 0.0
        
        if complexity > 0.7 or technical_level == "고급":
            return "고급"
        elif complexity > (0.4 + base_level) or technical_level == "중급":
            return "중급"
        else:
            return "초급"
    
    def preprocess_for_model(self, question: str) -> Dict:
        """모델 입력용 전처리 - 향상됨"""
        structure = self.analyze_question_structure(question)
        keywords = self.extract_keywords(question)
        difficulty = self.classify_question_difficulty(question)
        
        # 질문 정리 (더 보수적으로)
        cleaned_question = re.sub(r'\s+', ' ', question.strip())
        
        # 도메인별 특화 정보 추가
        domain = structure["domain_hints"][0] if structure["domain_hints"] else "일반"
        domain_context = self._get_domain_context(domain)
        
        return {
            "cleaned_question": cleaned_question,
            "structure": structure,
            "keywords": keywords,
            "difficulty": difficulty,
            "char_count": len(cleaned_question),
            "word_count": len(cleaned_question.split()),
            "domain": domain,
            "domain_context": domain_context,
            "is_reliable_classification": structure.get("mc_confidence", 0) >= 3
        }
    
    def _get_domain_context(self, domain: str) -> Dict:
        """도메인별 컨텍스트 정보"""
        domain_contexts = {
            "개인정보": {
                "key_laws": ["개인정보보호법"],
                "key_concepts": ["수집", "이용", "제공", "파기", "동의", "정보주체"],
                "authorities": ["개인정보보호위원회"]
            },
            "전자금융": {
                "key_laws": ["전자금융거래법"],
                "key_concepts": ["접근매체", "전자서명", "전자인증", "금융거래"],
                "authorities": ["금융위원회", "금융감독원"]
            },
            "사이버보안": {
                "key_laws": ["정보보안기본법"],
                "key_concepts": ["악성코드", "침입탐지", "취약점", "위협분석"],
                "authorities": ["KISA", "NCSC"]
            },
            "정보보안": {
                "key_laws": ["정보보안기본법", "개인정보보호법"],
                "key_concepts": ["정보보안관리체계", "접근통제", "암호화", "보안정책"],
                "authorities": ["KISA", "개인정보보호위원회"]
            }
        }
        
        return domain_contexts.get(domain, {
            "key_laws": [],
            "key_concepts": [],
            "authorities": []
        })
    
    def cleanup(self) -> None:
        """리소스 정리"""
        if self.debug_mode:
            print("데이터 처리기 정리 완료")


class DataProcessor:
    """기존 데이터 처리기 (하위 호환성)"""
    
    def __init__(self):
        self.real_processor = RealDataProcessor()
    
    def analyze_question_structure(self, question: str) -> Dict:
        return self.real_processor.analyze_question_structure(question)
    
    def _clean_korean_text(self, text: str) -> str:
        return self.real_processor._clean_korean_text(text)
    
    def cleanup(self) -> None:
        self.real_processor.cleanup()