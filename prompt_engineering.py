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
from typing import Dict, List, Optional
from knowledge_base import FinancialSecurityKnowledgeBase

class PromptEngineer:
    
    def __init__(self):
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.templates = self._build_templates()
        
        self.prompt_cache = {}
        self.max_cache_size = 300
        
        self.stats = {
            "cache_hits": 0,
            "template_usage": {},
            "domain_distribution": {}
        }
    
    def _build_templates(self) -> Dict[str, str]:
        templates = {}
        
        # 객관식 템플릿
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

        # 주관식 템플릿
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
        
        return templates
    
    def create_korean_reinforced_prompt(self, question: str, question_type: str) -> str:
        question_lower = question.lower()
        
        if question_type == "multiple_choice":
            # 특화된 도메인 템플릿 우선 적용
            if "금융투자업" in question_lower:
                if "소비자금융업" in question_lower or "보험중개업" in question_lower:
                    self.stats["template_usage"]["mc_financial"] = self.stats["template_usage"].get("mc_financial", 0) + 1
                    return self.templates["mc_financial"].format(question=question)
            
            if "위험" in question_lower and "관리" in question_lower:
                if "위험수용" in question_lower or "위험 수용" in question_lower:
                    self.stats["template_usage"]["mc_risk"] = self.stats["template_usage"].get("mc_risk", 0) + 1
                    return self.templates["mc_risk"].format(question=question)
            
            if "관리체계" in question_lower and "정책" in question_lower:
                if "경영진" in question_lower or "참여" in question_lower:
                    self.stats["template_usage"]["mc_management"] = self.stats["template_usage"].get("mc_management", 0) + 1
                    return self.templates["mc_management"].format(question=question)
            
            if "재해" in question_lower and "복구" in question_lower:
                if "개인정보" in question_lower and "파기" in question_lower:
                    self.stats["template_usage"]["mc_recovery"] = self.stats["template_usage"].get("mc_recovery", 0) + 1
                    return self.templates["mc_recovery"].format(question=question)
            
            if "트로이" in question_lower or "악성코드" in question_lower:
                self.stats["template_usage"]["mc_cyber_security"] = self.stats["template_usage"].get("mc_cyber_security", 0) + 1
                return self.templates["mc_cyber_security"].format(question=question)
            
            # 모두 포함 옵션 체크
            for choice_line in question.split('\n'):
                if re.match(r'^\s*[5]', choice_line):
                    if "모두" in choice_line or "전부" in choice_line:
                        self.stats["template_usage"]["mc_all_option"] = self.stats["template_usage"].get("mc_all_option", 0) + 1
                        return self.templates["mc_all_option"].format(question=question)
            
            # 부정형 문제 체크
            if any(neg in question_lower for neg in ["해당하지", "적절하지", "옳지", "틀린"]):
                negative_keywords = ["해당하지 않는", "적절하지 않은", "옳지 않은", "틀린"]
                keyword = next((k for k in negative_keywords if k in question), "해당하지 않는")
                self.stats["template_usage"]["mc_negative"] = self.stats["template_usage"].get("mc_negative", 0) + 1
                return self.templates["mc_negative"].format(question=question, keyword=keyword)
            
            # 기본 객관식 템플릿
            self.stats["template_usage"]["mc_direct"] = self.stats["template_usage"].get("mc_direct", 0) + 1
            return self.templates["mc_direct"].format(question=question)
            
        else:
            # 주관식 템플릿 선택
            if "트로이" in question_lower and any(word in question_lower for word in ["악성코드", "원격", "rat", "탐지"]):
                self.stats["template_usage"]["subj_trojan"] = self.stats["template_usage"].get("subj_trojan", 0) + 1
                return self.templates["subj_trojan"].format(question=question)
            elif "개인정보" in question_lower:
                self.stats["template_usage"]["subj_personal_info"] = self.stats["template_usage"].get("subj_personal_info", 0) + 1
                return self.templates["subj_personal_info"].format(question=question)
            elif "전자금융" in question_lower:
                self.stats["template_usage"]["subj_electronic"] = self.stats["template_usage"].get("subj_electronic", 0) + 1
                return self.templates["subj_electronic"].format(question=question)
            elif "위험" in question_lower and "관리" in question_lower:
                self.stats["template_usage"]["subj_risk_management"] = self.stats["template_usage"].get("subj_risk_management", 0) + 1
                return self.templates["subj_risk_management"].format(question=question)
            elif "관리체계" in question_lower:
                self.stats["template_usage"]["subj_management_system"] = self.stats["template_usage"].get("subj_management_system", 0) + 1
                return self.templates["subj_management_system"].format(question=question)
            elif "암호" in question_lower:
                self.stats["template_usage"]["subj_crypto"] = self.stats["template_usage"].get("subj_crypto", 0) + 1
                return self.templates["subj_crypto"].format(question=question)
            else:
                self.stats["template_usage"]["subj_enhanced"] = self.stats["template_usage"].get("subj_enhanced", 0) + 1
                return self.templates["subj_enhanced"].format(question=question)
    
    def create_prompt(self, question: str, question_type: str, analysis: Dict, structure: Dict) -> str:
        cache_key = hash(f"{question[:100]}{question_type}")
        if cache_key in self.prompt_cache:
            self.stats["cache_hits"] += 1
            return self.prompt_cache[cache_key]
        
        prompt = self.create_korean_reinforced_prompt(question, question_type)
        
        if len(self.prompt_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.prompt_cache))
            del self.prompt_cache[oldest_key]
        self.prompt_cache[cache_key] = prompt
        
        self._update_stats(analysis)
        
        return prompt
    
    def create_few_shot_prompt(self, question: str, question_type: str, analysis: Dict, num_examples: int = 1) -> str:
        prompt_parts = ["다음은 한국어 금융보안 문제 예시입니다.\n"]
        
        if question_type == "multiple_choice":
            question_lower = question.lower()
            
            if "금융투자업" in question_lower:
                prompt_parts.append("예시 문제: 금융투자업의 구분에 해당하지 않는 것은?")
                prompt_parts.append("정답: 1\n")
            elif "위험" in question_lower and "관리" in question_lower:
                prompt_parts.append("예시 문제: 위험 관리 계획 수립 시 고려해야 할 요소로 적절하지 않은 것은?")
                prompt_parts.append("정답: 2\n")
            else:
                prompt_parts.append("예시 문제: 개인정보의 정의로 가장 적절한 것은?")
                prompt_parts.append("정답: 2\n")
        else:
            if "트로이" in question:
                prompt_parts.append("예시 질문: 트로이 목마 기반 원격제어 악성코드의 특징과 탐지 지표를 설명하세요.")
                prompt_parts.append("답변: 트로이 목마는 정상 프로그램으로 위장한 악성코드로, 원격 접근 트로이 목마는 공격자가 감염된 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결, 시스템 리소스 사용 증가, 알 수 없는 프로세스 실행 등이 있습니다.\n")
        
        prompt_parts.append(f"현재 문제:\n{question}")
        prompt_parts.append("위 예시처럼 답하세요.")
        
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
    
    def _update_stats(self, analysis: Dict):
        domains = analysis.get("domain", ["일반"])
        for domain in domains:
            self.stats["domain_distribution"][domain] = self.stats["domain_distribution"].get(domain, 0) + 1
    
    def get_stats_report(self) -> Dict:
        total_prompts = sum(self.stats["template_usage"].values())
        
        return {
            "total_prompts": total_prompts,
            "cache_hit_rate": self.stats["cache_hits"] / max(total_prompts, 1),
            "template_usage": self.stats["template_usage"],
            "domain_distribution": self.stats["domain_distribution"]
        }
    
    def cleanup(self):
        if self.stats["template_usage"]:
            total = sum(self.stats["template_usage"].values())
            if total > 0:
                most_used = max(self.stats["template_usage"].items(), key=lambda x: x[1])
        
        self.prompt_cache.clear()