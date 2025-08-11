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
        self.templates = self._build_simple_templates()
        
        self.prompt_cache = {}
        self.max_cache_size = 200
        
        self.stats = {
            "cache_hits": 0,
            "template_usage": {},
            "domain_distribution": {}
        }
    
    def _build_simple_templates(self) -> Dict[str, str]:
        templates = {}
        
        templates["mc_basic"] = """{question}

위 문제의 정답 번호를 선택하세요 (1, 2, 3, 4, 5 중 하나)."""

        templates["mc_negative"] = """{question}

이 문제는 틀린 것, 해당하지 않는 것, 적절하지 않은 것을 찾는 문제입니다.
정답 번호를 선택하세요 (1, 2, 3, 4, 5 중 하나)."""

        templates["mc_financial"] = """{question}

금융투자업 분류:
- 금융투자업: 투자매매업, 투자중개업, 투자자문업, 투자일임업
- 금융투자업 아님: 소비자금융업, 보험중개업

정답 번호를 선택하세요 (1, 2, 3, 4, 5 중 하나)."""

        templates["subj_basic"] = """{question}

위 질문에 대해 한국어로 답변하세요.
법령과 규정에 따른 구체적인 설명을 포함하세요."""

        templates["subj_trojan"] = """{question}

트로이 목마의 특징과 탐지 방법에 대해 한국어로 설명하세요."""

        templates["subj_personal_info"] = """{question}

개인정보보호법에 따른 조치사항을 한국어로 설명하세요."""

        templates["subj_electronic"] = """{question}

전자금융거래법에 따른 안전성 확보 방안을 한국어로 설명하세요."""

        return templates
    
    def create_korean_reinforced_prompt(self, question: str, question_type: str) -> str:
        cache_key = hash(f"{question[:100]}{question_type}")
        if cache_key in self.prompt_cache:
            self.stats["cache_hits"] += 1
            return self.prompt_cache[cache_key]
        
        question_lower = question.lower()
        
        if question_type == "multiple_choice":
            if "금융투자업" in question_lower and ("소비자금융업" in question_lower or "보험중개업" in question_lower):
                template_key = "mc_financial"
                prompt = self.templates[template_key].format(question=question)
            elif any(neg in question_lower for neg in ["해당하지", "적절하지", "옳지", "틀린"]):
                template_key = "mc_negative"
                prompt = self.templates[template_key].format(question=question)
            else:
                template_key = "mc_basic"
                prompt = self.templates[template_key].format(question=question)
        else:
            if "트로이" in question_lower or "악성코드" in question_lower:
                template_key = "subj_trojan"
                prompt = self.templates[template_key].format(question=question)
            elif "개인정보" in question_lower:
                template_key = "subj_personal_info"
                prompt = self.templates[template_key].format(question=question)
            elif "전자금융" in question_lower:
                template_key = "subj_electronic"
                prompt = self.templates[template_key].format(question=question)
            else:
                template_key = "subj_basic"
                prompt = self.templates[template_key].format(question=question)
        
        self.stats["template_usage"][template_key] = self.stats["template_usage"].get(template_key, 0) + 1
        
        self._manage_cache()
        self.prompt_cache[cache_key] = prompt
        
        return prompt
    
    def _manage_cache(self):
        if len(self.prompt_cache) >= self.max_cache_size:
            keys_to_remove = list(self.prompt_cache.keys())[:self.max_cache_size // 3]
            for key in keys_to_remove:
                del self.prompt_cache[key]
    
    def create_prompt(self, question: str, question_type: str, analysis: Dict, structure: Dict) -> str:
        return self.create_korean_reinforced_prompt(question, question_type)
    
    def create_few_shot_prompt(self, question: str, question_type: str, analysis: Dict, num_examples: int = 1) -> str:
        prompt_parts = []
        
        if question_type == "multiple_choice":
            prompt_parts.append("예시: 개인정보의 정의로 가장 적절한 것은?")
            prompt_parts.append("정답: 2")
            prompt_parts.append("")
        else:
            prompt_parts.append("예시: 트로이 목마의 특징을 설명하세요.")
            prompt_parts.append("답변: 트로이 목마는 정상 프로그램으로 위장한 악성코드입니다.")
            prompt_parts.append("")
        
        prompt_parts.append(f"문제: {question}")
        
        if question_type == "multiple_choice":
            prompt_parts.append("정답:")
        else:
            prompt_parts.append("답변:")
        
        return "\n".join(prompt_parts)
    
    def optimize_for_model(self, prompt: str, model_name: str) -> str:
        if "solar" in model_name.lower():
            return f"### User:\n{prompt}\n\n### Assistant:\n"
        else:
            return prompt
    
    def get_stats_report(self) -> Dict:
        total_prompts = sum(self.stats["template_usage"].values())
        
        return {
            "total_prompts": total_prompts,
            "cache_hit_rate": self.stats["cache_hits"] / max(total_prompts, 1),
            "template_usage": self.stats["template_usage"],
            "domain_distribution": self.stats["domain_distribution"]
        }
    
    def cleanup(self):
        total_usage = sum(self.stats["template_usage"].values())
        if total_usage > 0:
            most_used = max(self.stats["template_usage"].items(), key=lambda x: x[1])
        
        self.prompt_cache.clear()