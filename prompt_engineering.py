# prompt_engineering.py

"""
프롬프트 엔지니어링
- 객관식/주관식 프롬프트 생성
- 도메인별 템플릿 관리
- 한국어 강화 프롬프트
- 패턴 기반 힌트 적용
"""

import re
import hashlib
import time
from typing import Dict, List, Optional, Tuple
from knowledge_base import FinancialSecurityKnowledgeBase

class PromptEngineer:
    
    def __init__(self):
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.templates = self._build_templates()
        self.adaptive_templates = self._build_adaptive_templates()
        
        self.prompt_cache = {}
        self.max_cache_size = 500
        
        self.stats = {
            "cache_hits": 0,
            "template_usage": {},
            "domain_distribution": {},
            "adaptive_selections": 0,
            "pattern_matches": 0,
            "successful_patterns": {}
        }
        
        self.pattern_success_tracking = {}
        self.template_performance = {}
        
    def _build_templates(self) -> Dict[str, str]:
        templates = {}
        
        templates["mc_direct"] = """### 문제
{question}

위 문제의 정답 번호만 출력하세요.
정답:"""

        templates["mc_basic"] = """당신은 한국의 금융보안 전문가입니다.

### 문제
{question}

### 답변 규칙
1. 반드시 1, 2, 3, 4, 5 중 하나만 선택
2. 정답 번호만 출력
3. 설명 없이 숫자만

정답:"""

        templates["mc_negative"] = """당신은 한국의 금융보안 전문가입니다.

### 문제
{question}

### 중요 힌트
이 문제는 '{keyword}' 문제입니다.
틀린 것, 해당하지 않는 것, 적절하지 않은 것을 찾으세요.

### 답변 규칙
1. 반드시 1, 2, 3, 4, 5 중 하나만 선택
2. 부정형 문제임을 명심
3. 정답 번호만 출력

정답:"""

        templates["mc_financial"] = """### 문제
{question}

### 금융업 분류 힌트
- 금융투자업: 투자매매업, 투자중개업, 투자자문업, 투자일임업
- 금융투자업 아님: 소비자금융업, 보험중개업

정답:"""

        templates["mc_risk"] = """### 문제
{question}

### 위험관리 힌트
- 위험관리 계획 수립 시 고려 요소: 대상, 기간, 수행인력, 대응전략
- 위험수용은 대응전략의 하나

정답:"""

        templates["mc_management"] = """### 문제
{question}

### 관리체계 힌트
- 정책수립 단계에서 가장 중요: 경영진의 참여와 지원
- 그 다음: 최고책임자 지정, 자원 할당

정답:"""

        templates["mc_recovery"] = """### 문제
{question}

### 재해복구 힌트
- 재해복구 계획 포함 사항: 복구절차, 비상연락체계, 복구목표시간
- 포함되지 않음: 개인정보 파기 절차

정답:"""

        templates["mc_cyber_security"] = """### 문제
{question}

### 사이버보안 힌트
- 트로이 목마: 정상 프로그램으로 위장한 악성코드
- RAT: 원격 접근 트로이 목마, 시스템 원격 제어
- 주요 탐지 지표: 비정상적 네트워크 연결, 시스템 리소스 증가

정답:"""

        templates["mc_all_option"] = """### 문제
{question}

### 힌트
마지막 선택지에 '모두' 또는 '전부'가 있는 경우 해당 번호 선택 가능성 높음

정답:"""

        templates["mc_enhanced"] = """당신은 한국의 금융보안 전문가입니다.

### 문제 분석
{question}

### 도메인: {domain}
### 특성: {characteristics}

### 답변 규칙
1. 반드시 1, 2, 3, 4, 5 중 하나만 선택
2. 도메인 지식을 활용한 논리적 추론
3. 정답 번호만 출력

정답:"""

        templates["subj_enhanced"] = """당신은 한국의 금융보안 전문가입니다.

### 중요 규칙
1. 반드시 순수 한국어로만 답변
2. 한자, 영어 등 외국어 절대 금지
3. 80-300자 내외로 답변
4. 전문적이고 명확한 한국어 사용

### 질문
{question}

### 답변 형식
관련 법령과 규정에 따라 구체적으로 설명하되,
순수 한국어만 사용하여 답변하세요.

답변:"""

        templates["subj_trojan"] = """당신은 한국의 사이버보안 전문가입니다.

### 중요 규칙
1. 반드시 순수 한국어로만 답변
2. 한자, 영어 등 외국어 절대 금지
3. 100-250자 내외로 답변

### 질문
{question}

### 답변 구조
1. 트로이 목마의 특징
2. 원격 접근 트로이 목마의 기능
3. 주요 탐지 지표

답변:"""

        templates["subj_personal_info"] = """당신은 한국의 개인정보보호 전문가입니다.

### 질문
{question}

### 답변 지침
개인정보보호법에 따른 구체적인 조치사항을 순수 한국어로 설명하세요.
80-250자 내외, 외국어 사용 금지

답변:"""

        templates["subj_electronic"] = """당신은 한국의 전자금융 전문가입니다.

### 질문
{question}

### 답변 지침
전자금융거래법에 따른 안전성 확보 방안을 순수 한국어로 설명하세요.
80-250자 내외, 외국어 사용 금지

답변:"""

        templates["subj_risk_management"] = """당신은 한국의 위험관리 전문가입니다.

### 질문
{question}

### 답변 지침
위험관리 체계와 대응전략을 순수 한국어로 설명하세요.
위험 식별, 평가, 대응, 모니터링 과정을 포함하여 설명하세요.
80-250자 내외, 외국어 사용 금지

답변:"""

        templates["subj_management_system"] = """당신은 한국의 관리체계 전문가입니다.

### 질문
{question}

### 답변 지침
관리체계 수립과 운영 방안을 순수 한국어로 설명하세요.
정책 수립, 조직 구성, 역할 분담, 지속적 개선을 포함하여 설명하세요.
80-250자 내외, 외국어 사용 금지

답변:"""

        templates["subj_crypto"] = """당신은 한국의 암호화 전문가입니다.

### 질문
{question}

### 답변 지침
암호화 기술과 키 관리 방안을 순수 한국어로 설명하세요.
대칭키, 공개키 암호화와 해시 함수 활용을 포함하여 설명하세요.
80-250자 내외, 외국어 사용 금지

답변:"""

        templates["subj_adaptive"] = """당신은 한국의 금융보안 전문가입니다.

### 중요 규칙
1. 반드시 순수 한국어로만 답변
2. 한자, 영어 등 외국어 절대 금지
3. 80-300자 내외로 답변

### 도메인: {domain}
### 문제 특성: {characteristics}

### 질문
{question}

### 답변 지침
{guidance}

답변:"""
        
        return templates
    
    def _build_adaptive_templates(self) -> Dict[str, Dict]:
        return {
            "mc_pattern_based": {
                "triggers": ["금융투자업", "위험관리", "관리체계", "재해복구", "개인정보", "전자금융"],
                "success_indicators": ["정확한 선택", "논리적 추론", "도메인 지식"],
                "template": """### 문제 분석
{question}

### 패턴 매칭 결과
- 도메인: {domain}
- 핵심 키워드: {keywords}
- 예상 답변 유형: {answer_type}

### 해결 전략
{strategy}

정답:"""
            },
            "subj_context_aware": {
                "triggers": ["설명하세요", "기술하세요", "서술하세요", "논하세요"],
                "success_indicators": ["한국어 품질", "전문성", "완성도"],
                "template": """### 맞춤형 전문가 답변

도메인: {domain}
문제 유형: {question_type}
복잡도: {complexity}

### 질문
{question}

### 전문가 답변 가이드
{expert_guidance}

### 중요사항
- 순수 한국어만 사용
- 전문적이고 정확한 설명
- 법령 및 규정 근거 제시

답변:"""
            },
            "mc_difficulty_adjusted": {
                "triggers": ["복잡한 구조", "다중 조건", "법령 참조"],
                "success_indicators": ["정확성", "추론 과정", "근거"],
                "template": """### 고급 문제 해결

문제 난이도: {difficulty}
예상 시간: {time_estimate}

### 문제
{question}

### 해결 접근법
{approach}

### 핵심 포인트
{key_points}

정답:"""
            }
        }
    
    def _analyze_question_patterns(self, question: str, structure: Dict) -> Dict:
        question_lower = question.lower()
        
        analysis = {
            "domain_strength": {},
            "pattern_matches": [],
            "complexity_factors": [],
            "special_characteristics": []
        }
        
        domain_keywords = {
            "금융투자업": ["금융투자업", "투자매매업", "소비자금융업", "보험중개업"],
            "위험관리": ["위험", "관리", "계획", "수립", "평가", "대응전략"],
            "관리체계": ["관리체계", "정책", "수립", "운영", "경영진"],
            "재해복구": ["재해", "복구", "계획", "비상", "백업"],
            "개인정보": ["개인정보", "정보주체", "보호법", "유출", "동의"],
            "전자금융": ["전자금융", "전자적", "장치", "접근매체", "거래"],
            "사이버보안": ["트로이", "악성코드", "해킹", "피싱", "멀웨어"],
            "암호화": ["암호화", "복호화", "키", "해시", "인증서"]
        }
        
        for domain, keywords in domain_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in question_lower)
            if matches > 0:
                strength = matches / len(keywords)
                analysis["domain_strength"][domain] = strength
                if strength > 0.3:
                    analysis["pattern_matches"].append(f"strong_{domain}")
                elif strength > 0.1:
                    analysis["pattern_matches"].append(f"weak_{domain}")
        
        if structure.get("has_negative", False):
            analysis["special_characteristics"].append("negative_question")
        
        if len(structure.get("choices", [])) >= 5:
            analysis["special_characteristics"].append("full_choice_set")
        
        if structure.get("has_all_option", False):
            analysis["special_characteristics"].append("all_option_present")
        
        complexity = structure.get("complexity_score", 0)
        if complexity > 0.7:
            analysis["complexity_factors"].append("high_complexity")
        elif complexity > 0.4:
            analysis["complexity_factors"].append("medium_complexity")
        else:
            analysis["complexity_factors"].append("low_complexity")
        
        if len(structure.get("technical_terms", [])) > 2:
            analysis["complexity_factors"].append("technical_heavy")
        
        if len(structure.get("legal_references", [])) > 0:
            analysis["complexity_factors"].append("legal_references")
        
        return analysis
    
    def _select_optimal_template(self, question: str, question_type: str, 
                                structure: Dict, analysis: Dict) -> Tuple[str, str]:
        
        question_lower = question.lower()
        
        if question_type == "multiple_choice":
            domain_strength = analysis["domain_strength"]
            pattern_matches = analysis["pattern_matches"]
            characteristics = analysis["special_characteristics"]
            
            if any("strong_금융투자업" in match for match in pattern_matches):
                if "소비자금융업" in question_lower or "보험중개업" in question_lower:
                    return "mc_financial", "domain_specialized"
            
            if any("strong_위험관리" in match for match in pattern_matches):
                if "위험수용" in question_lower or "위험 수용" in question_lower:
                    return "mc_risk", "domain_specialized"
            
            if any("strong_관리체계" in match for match in pattern_matches):
                if "경영진" in question_lower or "참여" in question_lower:
                    return "mc_management", "domain_specialized"
            
            if any("strong_재해복구" in match for match in pattern_matches):
                if "개인정보" in question_lower and "파기" in question_lower:
                    return "mc_recovery", "domain_specialized"
            
            if any("strong_사이버보안" in match for match in pattern_matches):
                return "mc_cyber_security", "domain_specialized"
            
            if "all_option_present" in characteristics:
                return "mc_all_option", "structural"
            
            if "negative_question" in characteristics:
                return "mc_negative", "structural"
            
            if "high_complexity" in analysis["complexity_factors"]:
                if len(domain_strength) > 1:
                    primary_domain = max(domain_strength.items(), key=lambda x: x[1])[0]
                    characteristics_str = ", ".join(characteristics)
                    return "mc_enhanced", f"adaptive_complex_{primary_domain}"
                else:
                    return "mc_difficulty_adjusted", "adaptive_complex"
            
            return "mc_direct", "default"
            
        else:
            domain_strength = analysis["domain_strength"]
            pattern_matches = analysis["pattern_matches"]
            
            if any("strong_사이버보안" in match for match in pattern_matches):
                if "트로이" in question_lower and any(word in question_lower for word in ["탐지", "원격", "rat"]):
                    return "subj_trojan", "domain_specialized"
            
            if any("strong_개인정보" in match for match in pattern_matches):
                return "subj_personal_info", "domain_specialized"
            
            if any("strong_전자금융" in match for match in pattern_matches):
                return "subj_electronic", "domain_specialized"
            
            if any("strong_위험관리" in match for match in pattern_matches):
                return "subj_risk_management", "domain_specialized"
            
            if any("strong_관리체계" in match for match in pattern_matches):
                return "subj_management_system", "domain_specialized"
            
            if any("strong_암호화" in match for match in pattern_matches):
                return "subj_crypto", "domain_specialized"
            
            if "high_complexity" in analysis["complexity_factors"] and len(domain_strength) > 0:
                return "subj_adaptive", "adaptive_complex"
            
            return "subj_enhanced", "default"
    
    def _generate_adaptive_content(self, template_key: str, question: str, 
                                  structure: Dict, analysis: Dict) -> Dict[str, str]:
        
        content = {}
        
        domain_strength = analysis["domain_strength"]
        primary_domain = max(domain_strength.items(), key=lambda x: x[1])[0] if domain_strength else "일반"
        
        content["domain"] = primary_domain
        
        characteristics = []
        if structure.get("has_negative", False):
            characteristics.append("부정형 문제")
        if structure.get("has_all_option", False):
            characteristics.append("전체 선택지 포함")
        if len(structure.get("technical_terms", [])) > 2:
            characteristics.append("기술 용어 집중")
        if len(structure.get("legal_references", [])) > 0:
            characteristics.append("법령 참조")
        
        content["characteristics"] = ", ".join(characteristics) if characteristics else "일반적 구조"
        
        if template_key == "mc_enhanced":
            content["strategy"] = self._generate_mc_strategy(primary_domain, analysis)
        elif template_key == "subj_adaptive":
            content["guidance"] = self._generate_subj_guidance(primary_domain, analysis)
            content["expert_guidance"] = self._generate_expert_guidance(primary_domain, structure)
        elif template_key == "mc_difficulty_adjusted":
            content["difficulty"] = "높음" if "high_complexity" in analysis.get("complexity_factors", []) else "보통"
            content["time_estimate"] = "30-45초"
            content["approach"] = self._generate_approach(analysis)
            content["key_points"] = self._generate_key_points(primary_domain)
        
        return content
    
    def _generate_mc_strategy(self, domain: str, analysis: Dict) -> str:
        strategies = {
            "금융투자업": "금융투자업 분류 기준을 명확히 구분하여 소비자금융업과 보험중개업은 제외",
            "위험관리": "위험 식별, 평가, 대응, 모니터링 단계별 요소를 체계적으로 분석",
            "관리체계": "정책 수립 단계에서 경영진 참여의 중요성을 우선 고려",
            "재해복구": "재해복구 계획의 핵심 구성요소와 개인정보 처리 절차의 구별",
            "개인정보": "개인정보보호법상 정의와 처리 원칙을 기준으로 판단",
            "전자금융": "전자금융거래법의 안전성 확보 조치와 이용자 보호 원칙 적용",
            "사이버보안": "악성코드 유형별 특성과 탐지 방법의 기술적 차이점 분석"
        }
        
        return strategies.get(domain, "문제의 핵심 키워드와 법령 근거를 바탕으로 논리적 추론")
    
    def _generate_subj_guidance(self, domain: str, analysis: Dict) -> str:
        guidance = {
            "개인정보": "개인정보보호법 조항을 근거로 구체적 조치사항 설명",
            "전자금융": "전자금융거래법에 따른 안전성 확보 방안과 이용자 보호 조치",
            "위험관리": "위험관리 프로세스의 단계별 활동과 통제 방안",
            "관리체계": "관리체계 구축의 핵심 요소와 운영 방안",
            "사이버보안": "사이버 위협의 특성과 대응 기술의 구체적 적용 방법",
            "암호화": "암호화 기술의 원리와 키 관리 체계의 실무적 구현"
        }
        
        return guidance.get(domain, "관련 법령과 기술 표준에 따른 체계적 설명")
    
    def _generate_expert_guidance(self, domain: str, structure: Dict) -> str:
        base_guidance = {
            "개인정보": "개인정보보호법 제29조 안전성확보조치를 중심으로 기술적, 관리적, 물리적 조치를 구분하여 설명",
            "전자금융": "전자금융거래법 제21조 안전성 확보 의무를 바탕으로 접근매체 관리와 거래 보안 조치",
            "위험관리": "ISO 31000 위험관리 원칙에 따른 위험 식별, 분석, 평가, 처리의 체계적 접근",
            "사이버보안": "사이버보안 위협 인텔리전스와 대응 기술의 통합적 활용 방안"
        }
        
        guidance = base_guidance.get(domain, "관련 분야 전문 지식과 실무 경험을 바탕으로 한 체계적 설명")
        
        if structure.get("is_procedural", False):
            guidance += " 단계별 절차와 각 단계의 핵심 활동을 순서대로 제시"
        
        if len(structure.get("legal_references", [])) > 0:
            guidance += " 관련 법령 조항의 구체적 근거와 실무 적용 방안"
        
        return guidance
    
    def _generate_approach(self, analysis: Dict) -> str:
        if "legal_references" in analysis.get("complexity_factors", []):
            return "법령 조항의 정확한 해석과 실무 적용 사례를 통한 체계적 분석"
        elif "technical_heavy" in analysis.get("complexity_factors", []):
            return "기술적 원리와 실무 구현 방법의 단계별 분해를 통한 논리적 접근"
        else:
            return "핵심 개념의 정의와 상호 관계를 바탕으로 한 체계적 분석"
    
    def _generate_key_points(self, domain: str) -> str:
        key_points = {
            "금융투자업": "업종 분류 기준, 인허가 요건, 업무 범위의 명확한 구분",
            "위험관리": "위험 유형별 특성, 평가 방법론, 대응 전략의 적절성",
            "개인정보": "정보주체 권리, 처리 목적과 방법, 안전성 확보조치의 구체적 내용",
            "전자금융": "전자적 장치의 정의, 접근매체 관리, 거래 보안 기술의 적용"
        }
        
        return key_points.get(domain, "핵심 개념의 정확한 이해와 실무 적용의 구체적 방법")
    
    def create_korean_reinforced_prompt(self, question: str, question_type: str) -> str:
        cache_key = hash(f"{question[:150]}{question_type}")
        if cache_key in self.prompt_cache:
            self.stats["cache_hits"] += 1
            return self.prompt_cache[cache_key]
        
        structure = self.knowledge_base.analyze_question(question)
        analysis = self._analyze_question_patterns(question, structure)
        
        template_key, selection_reason = self._select_optimal_template(
            question, question_type, structure, analysis
        )
        
        self.stats["template_usage"][template_key] = self.stats["template_usage"].get(template_key, 0) + 1
        
        if selection_reason.startswith("adaptive"):
            self.stats["adaptive_selections"] += 1
        
        if analysis["pattern_matches"]:
            self.stats["pattern_matches"] += 1
            for pattern in analysis["pattern_matches"]:
                self.stats["successful_patterns"][pattern] = self.stats["successful_patterns"].get(pattern, 0) + 1
        
        if template_key in ["mc_enhanced", "subj_adaptive", "mc_difficulty_adjusted"]:
            adaptive_content = self._generate_adaptive_content(template_key, question, structure, analysis)
            if template_key == "mc_enhanced":
                prompt = self.templates[template_key].format(
                    question=question,
                    domain=adaptive_content["domain"],
                    characteristics=adaptive_content["characteristics"]
                )
            elif template_key == "subj_adaptive":
                prompt = self.templates[template_key].format(
                    question=question,
                    domain=adaptive_content["domain"],
                    characteristics=adaptive_content["characteristics"],
                    guidance=adaptive_content["guidance"]
                )
            elif template_key == "mc_difficulty_adjusted":
                prompt = self.templates[template_key].format(
                    question=question,
                    difficulty=adaptive_content["difficulty"],
                    time_estimate=adaptive_content["time_estimate"],
                    approach=adaptive_content["approach"],
                    key_points=adaptive_content["key_points"]
                )
            else:
                prompt = self.templates[template_key].format(question=question)
        else:
            if template_key == "mc_negative":
                negative_keywords = ["해당하지 않는", "적절하지 않은", "옳지 않은", "틀린"]
                keyword = next((k for k in negative_keywords if k in question), "해당하지 않는")
                prompt = self.templates[template_key].format(question=question, keyword=keyword)
            else:
                prompt = self.templates[template_key].format(question=question)
        
        self._manage_cache()
        self.prompt_cache[cache_key] = prompt
        
        self._update_stats(analysis)
        
        return prompt
    
    def _manage_cache(self):
        if len(self.prompt_cache) >= self.max_cache_size:
            keys_to_remove = list(self.prompt_cache.keys())[:self.max_cache_size // 4]
            for key in keys_to_remove:
                del self.prompt_cache[key]
    
    def create_prompt(self, question: str, question_type: str, analysis: Dict, structure: Dict) -> str:
        return self.create_korean_reinforced_prompt(question, question_type)
    
    def create_few_shot_prompt(self, question: str, question_type: str, analysis: Dict, num_examples: int = 1) -> str:
        structure = self.knowledge_base.analyze_question(question)
        pattern_analysis = self._analyze_question_patterns(question, structure)
        
        prompt_parts = ["다음은 한국어 금융보안 문제 해결 예시입니다.\n"]
        
        if question_type == "multiple_choice":
            question_lower = question.lower()
            
            if any("strong_금융투자업" in match for match in pattern_analysis["pattern_matches"]):
                prompt_parts.append("예시 문제: 다음 중 금융투자업의 구분에 해당하지 않는 것은?")
                prompt_parts.append("해결 과정: 금융투자업은 투자매매업, 투자중개업, 투자자문업, 투자일임업을 포함하며, 소비자금융업과 보험중개업은 제외됩니다.")
                prompt_parts.append("정답: 1\n")
            elif any("strong_위험관리" in match for match in pattern_analysis["pattern_matches"]):
                prompt_parts.append("예시 문제: 위험 관리 계획 수립 시 고려해야 할 요소로 적절하지 않은 것은?")
                prompt_parts.append("해결 과정: 위험관리 계획은 대상, 기간, 수행인력, 대응전략을 포함하며, 위험수용은 대응전략의 하나입니다.")
                prompt_parts.append("정답: 2\n")
            else:
                prompt_parts.append("예시 문제: 개인정보의 정의로 가장 적절한 것은?")
                prompt_parts.append("해결 과정: 개인정보보호법 제2조에 따라 살아있는 개인에 관한 정보로서 개인을 알아볼 수 있는 정보입니다.")
                prompt_parts.append("정답: 2\n")
        else:
            if any("strong_사이버보안" in match for match in pattern_analysis["pattern_matches"]):
                prompt_parts.append("예시 질문: 트로이 목마 기반 원격제어 악성코드의 특징과 탐지 지표를 설명하세요.")
                prompt_parts.append("예시 답변: 트로이 목마는 정상 프로그램으로 위장한 악성코드로, 원격 접근 트로이 목마는 공격자가 감염된 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결, 시스템 리소스 사용 증가, 알 수 없는 프로세스 실행 등이 있습니다.\n")
            else:
                prompt_parts.append("예시 질문: 개인정보보호를 위한 안전성 확보조치에 대해 설명하세요.")
                prompt_parts.append("예시 답변: 개인정보보호법 제29조에 따라 기술적, 관리적, 물리적 안전성 확보조치를 수립하여 개인정보의 안전한 관리와 정보주체의 권리 보호를 위한 체계적인 조치가 필요합니다.\n")
        
        prompt_parts.append(f"현재 문제:\n{question}")
        prompt_parts.append("위 예시의 접근 방식을 참고하여 답하세요.")
        
        if question_type == "multiple_choice":
            prompt_parts.append("정답:")
        else:
            prompt_parts.append("답변:")
        
        return "\n".join(prompt_parts)
    
    def optimize_for_model(self, prompt: str, model_name: str) -> str:
        korean_prefix = "### 중요: 반드시 한국어로만 답변하세요 ###\n\n"
        
        if "solar" in model_name.lower():
            optimized = f"{korean_prefix}### User:\n{prompt}\n\n### Assistant:\n"
        elif "llama" in model_name.lower():
            optimized = f"{korean_prefix}<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            optimized = korean_prefix + prompt
        
        return optimized
    
    def track_template_performance(self, template_key: str, success: bool, confidence: float):
        if template_key not in self.template_performance:
            self.template_performance[template_key] = {
                "total_uses": 0,
                "successes": 0,
                "total_confidence": 0.0,
                "avg_confidence": 0.0
            }
        
        perf = self.template_performance[template_key]
        perf["total_uses"] += 1
        if success:
            perf["successes"] += 1
        perf["total_confidence"] += confidence
        perf["avg_confidence"] = perf["total_confidence"] / perf["total_uses"]
    
    def get_template_recommendations(self, question: str, question_type: str) -> List[Tuple[str, float]]:
        structure = self.knowledge_base.analyze_question(question)
        analysis = self._analyze_question_patterns(question, structure)
        
        recommendations = []
        
        for template_key in self.templates.keys():
            if question_type == "multiple_choice" and not template_key.startswith("mc_"):
                continue
            if question_type == "subjective" and not template_key.startswith("subj_"):
                continue
            
            score = 0.0
            
            if template_key in self.template_performance:
                perf = self.template_performance[template_key]
                if perf["total_uses"] > 5:
                    success_rate = perf["successes"] / perf["total_uses"]
                    score += success_rate * 0.6 + perf["avg_confidence"] * 0.4
            
            if template_key in self.stats["template_usage"]:
                usage_freq = self.stats["template_usage"][template_key]
                score += min(usage_freq / 100, 0.2)
            
            recommendations.append((template_key, score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:3]
    
    def _update_stats(self, analysis: Dict):
        domains = list(analysis["domain_strength"].keys())
        for domain in domains:
            self.stats["domain_distribution"][domain] = self.stats["domain_distribution"].get(domain, 0) + 1
    
    def get_stats_report(self) -> Dict:
        total_prompts = sum(self.stats["template_usage"].values())
        
        adaptive_rate = self.stats["adaptive_selections"] / max(total_prompts, 1)
        pattern_match_rate = self.stats["pattern_matches"] / max(total_prompts, 1)
        
        top_templates = sorted(self.stats["template_usage"].items(), key=lambda x: x[1], reverse=True)[:5]
        top_patterns = sorted(self.stats["successful_patterns"].items(), key=lambda x: x[1], reverse=True)[:5]
        
        performance_summary = {}
        for template, perf in self.template_performance.items():
            if perf["total_uses"] > 3:
                success_rate = perf["successes"] / perf["total_uses"]
                performance_summary[template] = {
                    "success_rate": success_rate,
                    "avg_confidence": perf["avg_confidence"],
                    "total_uses": perf["total_uses"]
                }
        
        return {
            "total_prompts": total_prompts,
            "cache_hit_rate": self.stats["cache_hits"] / max(total_prompts, 1),
            "adaptive_selection_rate": adaptive_rate,
            "pattern_match_rate": pattern_match_rate,
            "template_usage": dict(top_templates),
            "successful_patterns": dict(top_patterns),
            "template_performance": performance_summary,
            "domain_distribution": self.stats["domain_distribution"]
        }
    
    def cleanup(self):
        total_usage = sum(self.stats["template_usage"].values())
        adaptive_rate = self.stats["adaptive_selections"] / max(total_usage, 1)
        
        if total_usage > 0:
            most_used = max(self.stats["template_usage"].items(), key=lambda x: x[1])
            print(f"템플릿 사용: {most_used[0]} ({most_used[1]}회), 적응형 선택률: {adaptive_rate:.1%}")
        
        self.prompt_cache.clear()