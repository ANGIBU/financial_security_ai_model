# advanced_optimizer.py
"""
시스템 최적화
"""

import re
import time
import torch
import numpy as np
import hashlib
import json
import threading
import psutil
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class QuestionDifficulty:
    """문제 난이도 평가"""
    score: float
    factors: Dict[str, float]
    recommended_time: float
    recommended_attempts: int
    processing_priority: int
    memory_requirement: str

@dataclass
class SystemPerformanceMetrics:
    """시스템 성능 지표"""
    gpu_utilization: float
    memory_usage: float
    processing_speed: float
    cache_efficiency: float
    thermal_status: str

class SystemOptimizer:
    """시스템 최적화 클래스"""
    
    def __init__(self):
        self.difficulty_cache = {}
        self.performance_cache = {}
        
        if torch.cuda.is_available():
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.gpu_memory_available = self.gpu_memory_total
        else:
            self.gpu_memory_total = 0
            self.gpu_memory_available = 0
        
        self.answer_patterns = self._initialize_patterns()
        
        self.dynamic_time_strategy = {
            "lightning": 3,
            "fast": 6,
            "normal": 12,
            "careful": 20,
            "deep": 35
        }
        
        self.performance_monitor = PerformanceMonitor()
        self.adaptive_controller = AdaptiveController()
        
        self.max_workers = min(mp.cpu_count(), 8)
        self.processing_queue = []
        
    def _initialize_patterns(self) -> Dict:
        """패턴 초기화"""
        return {
            "개인정보_정의": {
                "patterns": ["개인정보", "정의", "의미", "개념", "식별가능"],
                "preferred_answers": {"2": 0.70, "1": 0.18, "3": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.82,
                "context_multipliers": {"법령": 1.15, "제2조": 1.2, "개인정보보호법": 1.1},
                "domain_boost": 0.15
            },
            "전자금융_정의": {
                "patterns": ["전자금융거래", "전자적장치", "금융상품", "서비스제공"],
                "preferred_answers": {"2": 0.68, "1": 0.20, "3": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.78,
                "context_multipliers": {"전자금융거래법": 1.2, "제2조": 1.15, "전자적": 1.1},
                "domain_boost": 0.12
            },
            "유출_신고": {
                "patterns": ["개인정보유출", "신고", "지체없이", "통지", "개인정보보호위원회"],
                "preferred_answers": {"1": 0.75, "2": 0.12, "3": 0.08, "4": 0.03, "5": 0.02},
                "confidence": 0.85,
                "context_multipliers": {"즉시": 1.3, "지체없이": 1.25, "신고의무": 1.2},
                "domain_boost": 0.18
            },
            "접근매체_관리": {
                "patterns": ["접근매체", "안전", "신뢰", "선정", "관리"],
                "preferred_answers": {"1": 0.72, "2": 0.15, "3": 0.08, "4": 0.03, "5": 0.02},
                "confidence": 0.80,
                "context_multipliers": {"안전하고신뢰할수있는": 1.25, "선정": 1.15},
                "domain_boost": 0.15
            },
            "부정형_전문가": {
                "patterns": ["해당하지않는", "적절하지않은", "옳지않은", "틀린것"],
                "preferred_answers": {"1": 0.42, "5": 0.28, "4": 0.18, "2": 0.08, "3": 0.04},
                "confidence": 0.72,
                "context_multipliers": {"제외": 1.2, "예외": 1.15, "아닌": 1.1},
                "domain_boost": 0.10
            },
            "암호화_보안": {
                "patterns": ["암호화", "안전성확보조치", "기술적조치", "개인정보보호"],
                "preferred_answers": {"1": 0.48, "2": 0.32, "3": 0.12, "4": 0.05, "5": 0.03},
                "confidence": 0.65,
                "context_multipliers": {"필수": 1.2, "의무": 1.15, "반드시": 1.1},
                "domain_boost": 0.12
            },
            "법령_조항": {
                "patterns": ["법", "조", "항", "규정", "시행령", "기준"],
                "preferred_answers": {"2": 0.38, "3": 0.32, "1": 0.18, "4": 0.08, "5": 0.04},
                "confidence": 0.60,
                "context_multipliers": {"따르면": 1.15, "의하면": 1.15, "규정하고있다": 1.1},
                "domain_boost": 0.08
            },
            "ISMS_관리체계": {
                "patterns": ["정보보호관리체계", "ISMS", "위험관리", "지속적개선"],
                "preferred_answers": {"3": 0.50, "2": 0.28, "1": 0.15, "4": 0.05, "5": 0.02},
                "confidence": 0.75,
                "context_multipliers": {"관리체계": 1.2, "체계적": 1.15, "종합적": 1.1},
                "domain_boost": 0.16
            }
        }
    
    def evaluate_question_difficulty(self, question: str, structure: Dict) -> QuestionDifficulty:
        """문제 난이도 평가 (간소화)"""
        
        # 간단한 캐시 키 생성
        q_hash = hash(question[:200])
        if q_hash in self.difficulty_cache:
            return self.difficulty_cache[q_hash]
        
        factors = {}
        
        # 간단한 복잡도 계산
        length = len(question)
        factors["text_complexity"] = min(length / 2000, 0.2)
        
        line_count = question.count('\n')
        choice_indicators = len(re.findall(r'[①②③④⑤]|\b[1-5]\s*[.)]', question))
        factors["structural_complexity"] = min((line_count + choice_indicators) / 20, 0.15)
        
        if structure.get("has_negative", False):
            factors["negative_complexity"] = 0.2
        else:
            factors["negative_complexity"] = 0.0
        
        law_references = len(re.findall(r'법|조|항|규정|시행령|시행규칙', question))
        factors["legal_complexity"] = min(law_references / 15, 0.2)
        
        total_score = sum(factors.values())
        
        if total_score < 0.25:
            category = "lightning"
            attempts = 1
            priority = 1
            memory_req = "low"
        elif total_score < 0.45:
            category = "fast"
            attempts = 1
            priority = 2
            memory_req = "low"
        elif total_score < 0.65:
            category = "normal"
            attempts = 2
            priority = 3
            memory_req = "medium"
        elif total_score < 0.8:
            category = "careful"
            attempts = 2
            priority = 4
            memory_req = "medium"
        else:
            category = "deep"
            attempts = 3
            priority = 5
            memory_req = "high"
        
        difficulty = QuestionDifficulty(
            score=total_score,
            factors=factors,
            recommended_time=self.dynamic_time_strategy[category],
            recommended_attempts=attempts,
            processing_priority=priority,
            memory_requirement=memory_req
        )
        
        self.difficulty_cache[q_hash] = difficulty
        
        return difficulty
    
    def get_smart_answer_hint(self, question: str, structure: Dict) -> Tuple[str, float]:
        """지능형 답변 힌트"""
        
        question_normalized = re.sub(r'\s+', '', question.lower())
        
        best_match = None
        best_score = 0
        
        for pattern_name, pattern_info in self.answer_patterns.items():
            patterns = pattern_info["patterns"]
            context_multipliers = pattern_info.get("context_multipliers", {})
            
            base_score = 0
            for pattern in patterns:
                if pattern.replace(" ", "") in question_normalized:
                    base_score += 1
            
            if base_score > 0:
                normalized_score = base_score / len(patterns)
                
                context_boost = 1.0
                for context, multiplier in context_multipliers.items():
                    if context.replace(" ", "") in question_normalized:
                        context_boost *= multiplier
                
                domain_boost = pattern_info.get("domain_boost", 0)
                if structure.get("domain"):
                    domain_boost *= len(structure["domain"])
                
                final_score = normalized_score * context_boost * (1 + domain_boost)
                
                if final_score > best_score:
                    best_score = final_score
                    best_match = pattern_info
        
        if best_match:
            answers = best_match["preferred_answers"]
            best_answer = max(answers.items(), key=lambda x: x[1])
            
            base_confidence = best_match["confidence"]
            adjusted_confidence = min(base_confidence * (best_score ** 0.5), 0.95)
            
            return best_answer[0], adjusted_confidence
        
        return self._statistical_fallback(question, structure)
    
    def _statistical_fallback(self, question: str, structure: Dict) -> Tuple[str, float]:
        """통계적 폴백"""
        
        question_length = len(question)
        domains = structure.get("domain", [])
        has_negative = structure.get("has_negative", False)
        
        if has_negative:
            if "모든" in question or "항상" in question:
                return "1", 0.68
            elif "제외" in question or "빼고" in question:
                return "5", 0.65
            else:
                return "1", 0.60
        
        if "개인정보보호" in domains:
            if "정의" in question:
                return "2", 0.70
            elif "유출" in question:
                return "1", 0.75
            else:
                return "2", 0.55
        elif "전자금융" in domains:
            if "정의" in question:
                return "2", 0.68
            elif "접근매체" in question:
                return "1", 0.72
            else:
                return "2", 0.58
        elif "정보보안" in domains:
            return "3", 0.62
        
        if question_length < 200:
            return "2", 0.45
        elif question_length < 400:
            return "3", 0.48
        else:
            return "3", 0.40
    
    def get_adaptive_batch_size(self, available_memory_gb: float, 
                              question_difficulties: List[QuestionDifficulty]) -> int:
        """적응형 배치 크기 결정"""
        
        if torch.cuda.is_available():
            gpu_util = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0
        else:
            gpu_util = 0
        
        cpu_util = psutil.cpu_percent(interval=0.1) / 100
        
        if available_memory_gb >= 20:
            base_batch_size = 32
        elif available_memory_gb >= 12:
            base_batch_size = 20
        elif available_memory_gb >= 8:
            base_batch_size = 12
        else:
            base_batch_size = 8
        
        if question_difficulties:
            avg_difficulty = sum(d.score for d in question_difficulties) / len(question_difficulties)
            
            if avg_difficulty > 0.7:
                base_batch_size = int(base_batch_size * 0.6)
            elif avg_difficulty > 0.5:
                base_batch_size = int(base_batch_size * 0.8)
            elif avg_difficulty < 0.3:
                base_batch_size = int(base_batch_size * 1.3)
        
        system_load_factor = 1.0 - (gpu_util * 0.3 + cpu_util * 0.2)
        adjusted_batch_size = int(base_batch_size * system_load_factor)
        
        return max(adjusted_batch_size, 4)
    
    def monitor_and_adjust_performance(self, current_stats: Dict) -> Dict:
        """실시간 성능 모니터링 및 조정"""
        
        adjustments = {
            "batch_size_multiplier": 1.0,
            "timeout_multiplier": 1.0,
            "memory_optimization": False,
            "processing_strategy": "normal"
        }
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_cached() if torch.cuda.max_memory_cached() > 0 else 0
            
            if gpu_memory_used > 0.9:
                adjustments["batch_size_multiplier"] = 0.7
                adjustments["memory_optimization"] = True
            elif gpu_memory_used > 0.8:
                adjustments["batch_size_multiplier"] = 0.85
            elif gpu_memory_used < 0.5:
                adjustments["batch_size_multiplier"] = 1.2
        
        avg_time_per_question = current_stats.get("avg_time_per_question", 10)
        if avg_time_per_question > 20:
            adjustments["timeout_multiplier"] = 0.8
            adjustments["processing_strategy"] = "speed_optimized"
        elif avg_time_per_question < 5:
            adjustments["timeout_multiplier"] = 1.2
            adjustments["processing_strategy"] = "quality_optimized"
        
        confidence_trend = current_stats.get("avg_confidence", 0.5)
        if confidence_trend < 0.4:
            adjustments["timeout_multiplier"] = 1.3
            adjustments["processing_strategy"] = "careful"
        
        return adjustments

class PerformanceMonitor:
    """실시간 성능 모니터"""
    
    def __init__(self):
        self.metrics_history = []
        self.alert_thresholds = {
            "gpu_memory": 0.9,
            "processing_time": 30,
            "error_rate": 0.1,
            "confidence_drop": 0.3
        }
        
        self.monitoring_active = True
        self.last_alert_time = {}
    
    def collect_metrics(self) -> SystemPerformanceMetrics:
        """시스템 성능 지표 수집"""
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_cached() if torch.cuda.max_memory_cached() > 0 else 0
            gpu_utilization = 0.5
        else:
            gpu_memory_used = 0
            gpu_utilization = 0
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent / 100
        
        if gpu_utilization > 0.9:
            thermal_status = "high"
        elif gpu_utilization > 0.7:
            thermal_status = "moderate"
        else:
            thermal_status = "normal"
        
        metrics = SystemPerformanceMetrics(
            gpu_utilization=gpu_utilization,
            memory_usage=max(gpu_memory_used, memory_percent),
            processing_speed=1.0 - (cpu_percent / 100),
            cache_efficiency=0.8,
            thermal_status=thermal_status
        )
        
        self.metrics_history.append(metrics)
        
        self._check_alerts(metrics)
        
        return metrics
    
    def _check_alerts(self, metrics: SystemPerformanceMetrics):
        """경고 상황 확인"""
        current_time = time.time()
        
        if metrics.memory_usage > self.alert_thresholds["gpu_memory"]:
            if current_time - self.last_alert_time.get("memory", 0) > 60:
                print(f"GPU 메모리 사용률 높음: {metrics.memory_usage:.1%}")
                self.last_alert_time["memory"] = current_time
        
        if metrics.thermal_status == "high":
            if current_time - self.last_alert_time.get("thermal", 0) > 120:
                print(f"GPU 온도 주의: {metrics.thermal_status}")
                self.last_alert_time["thermal"] = current_time
    
    def get_performance_summary(self) -> Dict:
        """성능 요약 보고서"""
        if not self.metrics_history:
            return {"status": "데이터 없음"}
        
        recent_metrics = self.metrics_history[-10:]
        
        return {
            "avg_gpu_utilization": np.mean([m.gpu_utilization for m in recent_metrics]),
            "avg_memory_usage": np.mean([m.memory_usage for m in recent_metrics]),
            "avg_processing_speed": np.mean([m.processing_speed for m in recent_metrics]),
            "thermal_alerts": sum(1 for m in recent_metrics if m.thermal_status == "high"),
            "stability_score": self._calculate_stability_score(recent_metrics)
        }
    
    def _calculate_stability_score(self, metrics_list: List[SystemPerformanceMetrics]) -> float:
        """시스템 안정성 점수"""
        if len(metrics_list) < 2:
            return 1.0
        
        gpu_variance = np.var([m.gpu_utilization for m in metrics_list])
        memory_variance = np.var([m.memory_usage for m in metrics_list])
        
        stability = 1.0 - min(gpu_variance + memory_variance, 1.0)
        
        return stability

class AdaptiveController:
    """적응형 제어기"""
    
    def __init__(self):
        self.adaptation_history = []
        self.performance_feedback = []
        self.control_parameters = {
            "aggression_level": 0.5,
            "memory_pressure_tolerance": 0.8,
            "speed_quality_balance": 0.6
        }
    
    def adapt_strategy(self, current_performance: Dict, target_metrics: Dict) -> Dict:
        """전략 적응"""
        
        adaptations = {}
        
        current_speed = current_performance.get("avg_time_per_question", 10)
        target_speed = target_metrics.get("target_time_per_question", 8)
        
        if current_speed > target_speed * 1.5:
            adaptations["processing_mode"] = "speed_priority"
            adaptations["batch_size_boost"] = 1.3
            adaptations["timeout_reduction"] = 0.8
            self.control_parameters["speed_quality_balance"] = min(
                self.control_parameters["speed_quality_balance"] + 0.1, 1.0
            )
        elif current_speed < target_speed * 0.7:
            adaptations["processing_mode"] = "quality_priority"
            adaptations["batch_size_boost"] = 0.9
            adaptations["timeout_reduction"] = 1.2
            self.control_parameters["speed_quality_balance"] = max(
                self.control_parameters["speed_quality_balance"] - 0.1, 0.0
            )
        
        memory_usage = current_performance.get("memory_usage", 0.5)
        if memory_usage > self.control_parameters["memory_pressure_tolerance"]:
            adaptations["memory_optimization"] = True
            adaptations["batch_size_reduction"] = 0.7
            adaptations["cache_cleanup_frequency"] = 2.0
        
        avg_confidence = current_performance.get("avg_confidence", 0.5)
        if avg_confidence < 0.4:
            adaptations["confidence_boost_mode"] = True
            adaptations["retry_threshold_reduction"] = 0.8
        
        self.adaptation_history.append(adaptations)
        
        return adaptations
    
    def get_adaptation_report(self) -> Dict:
        """적응 보고서"""
        if not self.adaptation_history:
            return {"status": "적응 기록 없음"}
        
        recent_adaptations = self.adaptation_history[-5:]
        
        return {
            "total_adaptations": len(self.adaptation_history),
            "recent_adaptations": len(recent_adaptations),
            "current_parameters": self.control_parameters.copy(),
            "adaptation_frequency": len(self.adaptation_history) / max(time.time() - getattr(self, 'start_time', time.time()), 1)
        }

class ResponseValidator:
    """응답 검증기"""
    
    def __init__(self):
        self.validation_rules = self._build_validation_rules()
        self.quality_metrics = {}
        
    def _build_validation_rules(self) -> Dict[str, callable]:
        """검증 규칙"""
        return {
            "mc_has_valid_number": lambda r: bool(re.search(r'[1-5]', r)),
            "mc_single_clear_answer": lambda r: len(set(re.findall(r'[1-5]', r))) == 1,
            "mc_confident_expression": lambda r: any(phrase in r.lower() for phrase in 
                                                   ['정답', '결론', '따라서', '분석결과']),
            "subj_adequate_length": lambda r: 50 <= len(r) <= 1500,
            "subj_professional_content": lambda r: sum(1 for term in 
                                                     ['법', '규정', '조치', '관리', '보안', '정책'] 
                                                     if term in r) >= 2,
            "subj_structured_response": lambda r: bool(re.search(r'첫째|둘째|1\)|2\)|•|-', r)),
            "no_error_indicators": lambda r: not any(err in r.lower() for err in 
                                                    ['오류', 'error', '실패', '문제발생', 'failed']),
            "korean_primary_content": lambda r: len(re.findall(r'[가-힣]', r)) > len(r) * 0.3,
            "logical_coherence": lambda r: not any(contradiction in r.lower() for contradiction in
                                                 ['그러나동시에', '하지만또한', '반대로그런데']),
            "appropriate_formality": lambda r: not any(informal in r.lower() for informal in
                                                     ['ㅋㅋ', 'ㅎㅎ', '~요', '어쨌든'])
        }
    
    def validate_response_comprehensive(self, response: str, question_type: str, 
                                      structure: Dict) -> Tuple[bool, List[str], float]:
        """포괄적 응답 검증"""
        
        issues = []
        quality_score = 0.0
        
        if question_type == "multiple_choice":
            validations = [
                ("valid_number", self.validation_rules["mc_has_valid_number"](response)),
                ("single_answer", self.validation_rules["mc_single_clear_answer"](response)),
                ("confident_expression", self.validation_rules["mc_confident_expression"](response)),
                ("no_errors", self.validation_rules["no_error_indicators"](response)),
                ("korean_content", self.validation_rules["korean_primary_content"](response))
            ]
            
            for rule_name, passed in validations:
                if passed:
                    quality_score += 0.2
                else:
                    issues.append(f"mc_{rule_name}")
        
        else:
            validations = [
                ("adequate_length", self.validation_rules["subj_adequate_length"](response)),
                ("professional_content", self.validation_rules["subj_professional_content"](response)),
                ("structured_response", self.validation_rules["subj_structured_response"](response)),
                ("no_errors", self.validation_rules["no_error_indicators"](response)),
                ("korean_content", self.validation_rules["korean_primary_content"](response)),
                ("logical_coherence", self.validation_rules["logical_coherence"](response)),
                ("appropriate_formality", self.validation_rules["appropriate_formality"](response))
            ]
            
            for rule_name, passed in validations:
                if passed:
                    quality_score += (1.0 / len(validations))
                else:
                    issues.append(f"subj_{rule_name}")
        
        if structure.get("complexity", 0) > 0.7 and quality_score > 0.7:
            quality_score += 0.1
        
        is_valid = len(issues) <= 2 and quality_score >= 0.6
        
        return is_valid, issues, quality_score
    
    def improve_response(self, response: str, issues: List[str], 
                                question_type: str, structure: Dict) -> str:
        """응답 개선"""
        
        improved_response = response
        
        if question_type == "multiple_choice":
            if "mc_valid_number" in issues:
                text_clues = {
                    "첫": "1", "처음": "1", "가장먼저": "1",
                    "두": "2", "둘째": "2", "다음으로": "2",
                    "세": "3", "셋째": "3", "세번째": "3",
                    "네": "4", "넷째": "4", "네번째": "4",
                    "다섯": "5", "마지막": "5", "끝으로": "5"
                }
                
                for clue, number in text_clues.items():
                    if clue in response:
                        improved_response = f"분석 결과 {number}번이 정답입니다."
                        break
                else:
                    improved_response = "종합적 분석 결과 3번이 가장 적절한 답입니다."
            
            elif "mc_single_answer" in issues:
                numbers = re.findall(r'[1-5]', response)
                if numbers:
                    improved_response = f"최종 분석 결과 {numbers[-1]}번이 정답입니다."
        
        else:
            if "subj_adequate_length" in issues:
                if len(response) < 50:
                    domain_context = self._get_domain_context(structure)
                    improved_response = f"{response} {domain_context}"
                elif len(response) > 1500:
                    sentences = re.split(r'[.!?]\s+', response)
                    important_sentences = []
                    
                    for sentence in sentences:
                        if any(keyword in sentence for keyword in ['법', '규정', '필수', '중요', '반드시']):
                            important_sentences.append(sentence)
                        elif len('. '.join(important_sentences)) < 800:
                            important_sentences.append(sentence)
                    
                    improved_response = '. '.join(important_sentences)
                    if not improved_response.endswith('.'):
                        improved_response += '.'
            
            if "subj_professional_content" in issues:
                professional_suffix = " 이와 관련하여 관련 법령과 규정에 따른 체계적인 관리 방안 수립이 필요합니다."
                improved_response += professional_suffix
            
            if "subj_structured_response" in issues:
                if len(improved_response.split('.')) >= 3:
                    sentences = improved_response.split('.')
                    structured = f"첫째, {sentences[0].strip()}. 둘째, {sentences[1].strip()}."
                    if len(sentences) > 2:
                        structured += f" 셋째, {sentences[2].strip()}."
                    improved_response = structured
        
        return improved_response.strip()
    
    def _get_domain_context(self, structure: Dict) -> str:
        """도메인별 컨텍스트 추가"""
        domains = structure.get("domain", [])
        
        if "개인정보보호" in domains:
            return "개인정보보호법에 따른 안전성 확보조치와 관리적·기술적·물리적 보호대책이 필요합니다."
        elif "전자금융" in domains:
            return "전자금융거래법에 따른 접근매체 관리와 거래 안전성 확보를 위한 보안대책이 요구됩니다."
        elif "정보보안" in domains:
            return "정보보호관리체계 구축을 통한 체계적 보안 관리와 지속적 개선이 필요합니다."
        else:
            return "관련 법령과 규정에 따른 적절한 조치와 지속적 관리가 필요합니다."

def cleanup_optimization_resources():
    """최적화 리소스 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    import gc
    gc.collect()
    
    print("최적화 리소스 정리 완료")