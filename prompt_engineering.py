# prompt_engineering.py

import re
import hashlib
from typing import Dict, List, Optional
from knowledge_base import FinancialSecurityKnowledgeBase

class PromptEngineer:
    
    def __init__(self):
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.templates = self._build_korean_templates()
        self.examples = self._build_korean_examples()
        
        self.prompt_cache = {}
        self.template_cache = {}
        self.max_cache_size = 300
        
        self.stats = {
            "cache_hits": 0,
            "template_usage": {},
            "domain_distribution": {}
        }
    
    def _build_korean_templates(self) -> Dict[str, str]:
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

        templates["mc_all_option"] = """### 문제
{question}

### 힌트
마지막 선택지에 '모두' 또는 '전부'가 있는 경우 해당 번호 선택 가능성 높음

정답:"""

        templates["mc_cyber_security"] = """### 문제
{question}

### 사이버보안 힌트
- 트로이 목마: 정상 프로그램으로 위장한 악성코드
- RAT: 원격 접근 트로이 목마, 시스템 원격 제어
- 주요 탐지 지표: 비정상적 네트워크 연결, 시스템 리소스 증가

정답:"""

        templates["mc_encryption"] = """### 문제
{question}

### 암호화 힌트
- 대칭키 암호화: 빠른 처리, 같은 키 사용
- 공개키 암호화: 안전한 키 교환, 디지털 서명
- 해시 함수: 무결성 검증, 단방향 암호화

정답:"""

        templates["mc_access_control"] = """### 문제
{question}

### 접근제어 힌트
- 접근매체: 안전하고 신뢰할 수 있어야 함
- 다중인증: 2개 이상 인증요소 조합
- 생체인증: 지문, 홍채, 얼굴 인식

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

### 용어 변환
- Trojan → 트로이 목마
- RAT → 원격 접근 트로이 목마
- Remote → 원격
- Access → 접근
- Malware → 악성코드

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

        templates["subj_incident_response"] = """당신은 한국의 사고대응 전문가입니다.

### 질문
{question}

### 답변 지침
침해사고 대응 절차와 복구 방안을 순수 한국어로 설명하세요.
사고 탐지, 분석, 대응, 복구, 사후관리 단계를 포함하여 설명하세요.
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

        templates["subj_law_compliance"] = """당신은 한국의 금융법규 준수 전문가입니다.

### 질문
{question}

### 답변 지침
관련 법령의 준수 사항과 의무 사항을 순수 한국어로 설명하세요.
법적 근거와 구체적 조치 방안을 포함하여 설명하세요.
80-250자 내외, 외국어 사용 금지

답변:"""
        
        return templates
    
    def _build_korean_examples(self) -> Dict[str, Dict]:
        examples = {
            "mc_financial": {
                "question": "금융투자업의 구분에 해당하지 않는 것은?",
                "answer": "1",
                "reasoning": "소비자금융업은 금융투자업이 아님"
            },
            "mc_risk": {
                "question": "위험 관리 계획 수립 시 고려해야 할 요소로 적절하지 않은 것은?",
                "answer": "2",
                "reasoning": "위험 수용은 대응 전략의 하나"
            },
            "mc_management": {
                "question": "관리체계 수립 시 정책수립 단계에서 가장 중요한 것은?",
                "answer": "2",
                "reasoning": "경영진의 참여와 지원이 가장 중요"
            },
            "mc_recovery": {
                "question": "재해복구 계획 수립 시 고려사항으로 옳지 않은 것은?",
                "answer": "3",
                "reasoning": "개인정보 파기 절차는 재해복구와 직접 관련 없음"
            },
            "mc_personal": {
                "question": "개인정보의 정의로 가장 적절한 것은?",
                "answer": "2",
                "reasoning": "살아있는 개인에 관한 정보로서 개인을 알아볼 수 있는 정보"
            },
            "mc_electronic": {
                "question": "전자금융거래의 정의로 가장 적절한 것은?",
                "answer": "2",
                "reasoning": "전자적 장치를 통하여 금융상품과 서비스를 제공하고 이용하는 거래"
            },
            "subj_trojan": {
                "question": "트로이 목마 기반 원격제어 악성코드의 특징과 탐지 지표를 설명하세요.",
                "answer": "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 원격 접근 트로이 목마는 공격자가 감염된 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결, 시스템 리소스 사용 증가, 알 수 없는 프로세스 실행, 방화벽 규칙 변경 등이 있습니다."
            }
        }
        return examples
    
    def create_prompt(self, question: str, question_type: str, 
                     analysis: Dict, structure: Dict) -> str:
        
        cache_key = hash(f"{question[:100]}{question_type}")
        if cache_key in self.prompt_cache:
            self.stats["cache_hits"] += 1
            return self.prompt_cache[cache_key]
        
        if question_type == "multiple_choice":
            prompt = self._create_mc_prompt_enhanced(question, analysis, structure)
        else:
            prompt = self._create_subj_prompt_enhanced(question, analysis, structure)
        
        if len(self.prompt_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.prompt_cache))
            del self.prompt_cache[oldest_key]
        self.prompt_cache[cache_key] = prompt
        
        self._update_stats(analysis)
        
        return prompt
    
    def _create_mc_prompt_enhanced(self, question: str, analysis: Dict, structure: Dict) -> str:
        question_lower = question.lower()
        
        if "금융투자업" in question_lower:
            if "소비자금융업" in question_lower or "보험중개업" in question_lower:
                prompt = self.templates["mc_financial"].format(question=question)
                self.stats["template_usage"]["mc_financial"] = self.stats["template_usage"].get("mc_financial", 0) + 1
                return prompt
        
        if "위험" in question_lower and "관리" in question_lower and "계획" in question_lower:
            if "위험수용" in question_lower or "위험 수용" in question_lower:
                prompt = self.templates["mc_risk"].format(question=question)
                self.stats["template_usage"]["mc_risk"] = self.stats["template_usage"].get("mc_risk", 0) + 1
                return prompt
        
        if "관리체계" in question_lower and "정책" in question_lower:
            if "경영진" in question_lower or "가장중요" in question_lower:
                prompt = self.templates["mc_management"].format(question=question)
                self.stats["template_usage"]["mc_management"] = self.stats["template_usage"].get("mc_management", 0) + 1
                return prompt
        
        if "재해복구" in question_lower or "재해 복구" in question_lower:
            if "개인정보파기" in question_lower or "개인정보 파기" in question_lower:
                prompt = self.templates["mc_recovery"].format(question=question)
                self.stats["template_usage"]["mc_recovery"] = self.stats["template_usage"].get("mc_recovery", 0) + 1
                return prompt
        
        if "트로이" in question_lower or "악성코드" in question_lower or "원격" in question_lower:
            prompt = self.templates["mc_cyber_security"].format(question=question)
            self.stats["template_usage"]["mc_cyber_security"] = self.stats["template_usage"].get("mc_cyber_security", 0) + 1
            return prompt
        
        if "암호화" in question_lower or "암호" in question_lower or "키관리" in question_lower:
            prompt = self.templates["mc_encryption"].format(question=question)
            self.stats["template_usage"]["mc_encryption"] = self.stats["template_usage"].get("mc_encryption", 0) + 1
            return prompt
        
        if "접근매체" in question_lower or "접근제어" in question_lower or "다중인증" in question_lower:
            prompt = self.templates["mc_access_control"].format(question=question)
            self.stats["template_usage"]["mc_access_control"] = self.stats["template_usage"].get("mc_access_control", 0) + 1
            return prompt
        
        if structure.get("has_all_option", False):
            prompt = self.templates["mc_all_option"].format(question=question)
            self.stats["template_usage"]["mc_all_option"] = self.stats["template_usage"].get("mc_all_option", 0) + 1
            return prompt
        
        if structure.get("has_negative", False):
            negative_keywords = ["해당하지 않는", "적절하지 않은", "옳지 않은", "틀린"]
            keyword = next((k for k in negative_keywords if k in question), "해당하지 않는")
            
            prompt = self.templates["mc_negative"].format(
                question=question,
                keyword=keyword
            )
            self.stats["template_usage"]["mc_negative"] = self.stats["template_usage"].get("mc_negative", 0) + 1
        else:
            prompt = self.templates["mc_direct"].format(question=question)
            self.stats["template_usage"]["mc_direct"] = self.stats["template_usage"].get("mc_direct", 0) + 1
        
        return prompt
    
    def _create_subj_prompt_enhanced(self, question: str, analysis: Dict, structure: Dict) -> str:
        question_lower = question.lower()
        domains = analysis.get("domain", [])
        
        if "트로이" in question_lower and ("악성코드" in question_lower or "원격" in question_lower or "탐지" in question_lower):
            prompt = self.templates["subj_trojan"].format(question=question)
            self.stats["template_usage"]["subj_trojan"] = self.stats["template_usage"].get("subj_trojan", 0) + 1
        elif "개인정보보호" in domains or "개인정보" in question_lower:
            prompt = self.templates["subj_personal_info"].format(question=question)
            self.stats["template_usage"]["subj_personal_info"] = self.stats["template_usage"].get("subj_personal_info", 0) + 1
        elif "전자금융" in domains or "전자금융" in question_lower:
            prompt = self.templates["subj_electronic"].format(question=question)
            self.stats["template_usage"]["subj_electronic"] = self.stats["template_usage"].get("subj_electronic", 0) + 1
        elif "위험관리" in domains or ("위험" in question_lower and "관리" in question_lower):
            prompt = self.templates["subj_risk_management"].format(question=question)
            self.stats["template_usage"]["subj_risk_management"] = self.stats["template_usage"].get("subj_risk_management", 0) + 1
        elif "관리체계" in domains or ("관리체계" in question_lower and "정책" in question_lower):
            prompt = self.templates["subj_management_system"].format(question=question)
            self.stats["template_usage"]["subj_management_system"] = self.stats["template_usage"].get("subj_management_system", 0) + 1
        elif "사고대응" in domains or ("사고" in question_lower and ("대응" in question_lower or "복구" in question_lower)):
            prompt = self.templates["subj_incident_response"].format(question=question)
            self.stats["template_usage"]["subj_incident_response"] = self.stats["template_usage"].get("subj_incident_response", 0) + 1
        elif "암호화" in domains or ("암호" in question_lower and "키" in question_lower):
            prompt = self.templates["subj_crypto"].format(question=question)
            self.stats["template_usage"]["subj_crypto"] = self.stats["template_usage"].get("subj_crypto", 0) + 1
        elif "법령" in question_lower or "규정" in question_lower or "의무" in question_lower:
            prompt = self.templates["subj_law_compliance"].format(question=question)
            self.stats["template_usage"]["subj_law_compliance"] = self.stats["template_usage"].get("subj_law_compliance", 0) + 1
        else:
            prompt = self.templates["subj_enhanced"].format(question=question)
            self.stats["template_usage"]["subj_enhanced"] = self.stats["template_usage"].get("subj_enhanced", 0) + 1
        
        return prompt
    
    def create_korean_reinforced_prompt(self, question: str, question_type: str) -> str:
        question_lower = question.lower()
        
        if question_type == "multiple_choice":
            if "금융투자업" in question_lower:
                if "소비자금융업" in question_lower or "보험중개업" in question_lower:
                    return self.templates["mc_financial"].format(question=question)
            
            if "위험" in question_lower and "관리" in question_lower:
                if "위험수용" in question_lower or "위험 수용" in question_lower:
                    return self.templates["mc_risk"].format(question=question)
            
            if "관리체계" in question_lower and "정책" in question_lower:
                if "경영진" in question_lower or "참여" in question_lower:
                    return self.templates["mc_management"].format(question=question)
            
            if "재해" in question_lower and "복구" in question_lower:
                if "개인정보" in question_lower and "파기" in question_lower:
                    return self.templates["mc_recovery"].format(question=question)
            
            if "트로이" in question_lower or "악성코드" in question_lower:
                return self.templates["mc_cyber_security"].format(question=question)
            
            if "암호화" in question_lower or "암호" in question_lower:
                return self.templates["mc_encryption"].format(question=question)
            
            if "접근매체" in question_lower or "접근제어" in question_lower:
                return self.templates["mc_access_control"].format(question=question)
            
            for choice_line in question.split('\n'):
                if re.match(r'^\s*[5]', choice_line):
                    if "모두" in choice_line or "전부" in choice_line:
                        return self.templates["mc_all_option"].format(question=question)
            
            if any(neg in question_lower for neg in ["해당하지", "적절하지", "옳지", "틀린"]):
                negative_keywords = ["해당하지 않는", "적절하지 않은", "옳지 않은", "틀린"]
                keyword = next((k for k in negative_keywords if k in question), "해당하지 않는")
                return self.templates["mc_negative"].format(question=question, keyword=keyword)
            
            return self.templates["mc_direct"].format(question=question)
            
        else:
            if "트로이" in question_lower and any(word in question_lower for word in ["악성코드", "원격", "rat", "탐지"]):
                return self.templates["subj_trojan"].format(question=question)
            elif "개인정보" in question_lower:
                return self.templates["subj_personal_info"].format(question=question)
            elif "전자금융" in question_lower:
                return self.templates["subj_electronic"].format(question=question)
            elif "위험" in question_lower and "관리" in question_lower:
                return self.templates["subj_risk_management"].format(question=question)
            elif "관리체계" in question_lower:
                return self.templates["subj_management_system"].format(question=question)
            elif "사고" in question_lower and ("대응" in question_lower or "복구" in question_lower):
                return self.templates["subj_incident_response"].format(question=question)
            elif "암호" in question_lower:
                return self.templates["subj_crypto"].format(question=question)
            elif "법령" in question_lower or "규정" in question_lower:
                return self.templates["subj_law_compliance"].format(question=question)
            else:
                return self.templates["subj_enhanced"].format(question=question)
    
    def create_few_shot_prompt(self, question: str, question_type: str,
                             analysis: Dict, num_examples: int = 1) -> str:
        
        prompt_parts = ["다음은 한국어 금융보안 문제 예시입니다.\n"]
        
        if question_type == "multiple_choice":
            question_lower = question.lower()
            
            if "금융투자업" in question_lower:
                example = self.examples["mc_financial"]
            elif "위험" in question_lower and "관리" in question_lower:
                example = self.examples["mc_risk"]
            elif "관리체계" in question_lower:
                example = self.examples["mc_management"]
            elif "재해복구" in question_lower:
                example = self.examples["mc_recovery"]
            elif "개인정보" in question_lower and "정의" in question_lower:
                example = self.examples["mc_personal"]
            elif "전자금융" in question_lower and "정의" in question_lower:
                example = self.examples["mc_electronic"]
            else:
                example = self.examples["mc_financial"]
            
            prompt_parts.append(f"예시 문제: {example['question']}")
            prompt_parts.append(f"정답: {example['answer']}\n")
        else:
            if "트로이" in question:
                example = self.examples["subj_trojan"]
                prompt_parts.append(f"예시 질문: {example['question']}")
                prompt_parts.append(f"답변: {example['answer']}\n")
        
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
        self.template_cache.clear()