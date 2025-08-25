# statistics_manager.py

import time
import gc
import psutil
from typing import Dict, List
from datetime import datetime
from pathlib import Path
from config import LOG_DIR


class StatisticsManager:
    """실행 통계 관리"""
    
    def __init__(self, log_type: str = "inference"):
        LOG_DIR.mkdir(exist_ok=True)
        if log_type == "test":
            self.log_file = LOG_DIR / "test.txt"
        else:
            self.log_file = LOG_DIR / "inference.txt"
        
        self.start_time = None
        self.processing_times = []
        self.domain_stats = {}
        self.method_stats = {}
        self.error_stats = {}
        self.template_usage_stats = {}
        self.memory_snapshots = []
        
    def start_session(self):
        """세션 시작"""
        self.start_time = time.time()
        self._log_system_info()
        
    def record_question_processing(self, processing_time: float, domain: str, method: str, 
                                  question_type: str, success: bool, error_type: str = None):
        """문항 처리 기록"""
        self.processing_times.append(processing_time)
        
        # 도메인 통계
        if domain not in self.domain_stats:
            self.domain_stats[domain] = {"count": 0, "avg_time": 0, "total_time": 0}
        self.domain_stats[domain]["count"] += 1
        self.domain_stats[domain]["total_time"] += processing_time
        self.domain_stats[domain]["avg_time"] = self.domain_stats[domain]["total_time"] / self.domain_stats[domain]["count"]
        
        # 처리 방법 통계
        if method not in self.method_stats:
            self.method_stats[method] = {"count": 0, "success_count": 0, "avg_time": 0, "total_time": 0}
        self.method_stats[method]["count"] += 1
        self.method_stats[method]["total_time"] += processing_time
        self.method_stats[method]["avg_time"] = self.method_stats[method]["total_time"] / self.method_stats[method]["count"]
        
        if success:
            self.method_stats[method]["success_count"] += 1
            
        # 오류 통계
        if error_type and not success:
            if error_type not in self.error_stats:
                self.error_stats[error_type] = 0
            self.error_stats[error_type] += 1
    
    def record_template_usage(self, template_type: str, usage_count: int = 1):
        """템플릿 사용 통계"""
        if template_type not in self.template_usage_stats:
            self.template_usage_stats[template_type] = 0
        self.template_usage_stats[template_type] += usage_count
    
    def record_memory_snapshot(self):
        """메모리 상태 기록"""
        try:
            memory_info = psutil.virtual_memory()
            self.memory_snapshots.append({
                "timestamp": time.time(),
                "used_percent": memory_info.percent,
                "available_mb": memory_info.available / 1024 / 1024
            })
        except:
            pass
    
    def generate_final_statistics(self, learning_data: Dict) -> Dict:
        """최종 통계 생성"""
        total_time = time.time() - self.start_time if self.start_time else 0
        total_questions = len(self.processing_times)
        
        # 기본 통계
        stats = {
            "execution_summary": {
                "total_time_seconds": round(total_time, 2),
                "total_time_minutes": round(total_time / 60, 2),
                "total_questions": total_questions,
                "avg_processing_time": round(sum(self.processing_times) / len(self.processing_times), 3) if self.processing_times else 0,
                "questions_per_minute": round((total_questions / (total_time / 60)), 2) if total_time > 0 else 0
            },
            "domain_analysis": self._analyze_domains(),
            "method_analysis": self._analyze_methods(),
            "performance_metrics": self._calculate_performance_metrics(),
            "learning_metrics": self._analyze_learning_data(learning_data),
            "system_metrics": self._analyze_system_metrics(),
            "error_analysis": self.error_stats if self.error_stats else {},
            "template_analysis": self.template_usage_stats if self.template_usage_stats else {}
        }
        
        self._write_detailed_log(stats)
        return stats
    
    def _log_system_info(self):
        """시스템 정보 기록"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] === 세션 시작 ===\n")
                f.write(f"[{timestamp}] 시스템 메모리: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB\n")
                f.write(f"[{timestamp}] CPU 코어: {psutil.cpu_count()}개\n")
        except:
            pass
    
    def _analyze_domains(self) -> Dict:
        """도메인 분석"""
        if not self.domain_stats:
            return {}
            
        total_questions = sum(domain["count"] for domain in self.domain_stats.values())
        
        domain_analysis = {}
        for domain, stats in self.domain_stats.items():
            domain_analysis[domain] = {
                "question_count": stats["count"],
                "percentage": round((stats["count"] / total_questions) * 100, 1),
                "avg_processing_time": round(stats["avg_time"], 3),
                "total_processing_time": round(stats["total_time"], 3)
            }
        
        return domain_analysis
    
    def _analyze_methods(self) -> Dict:
        """처리 방법 분석"""
        if not self.method_stats:
            return {}
            
        method_analysis = {}
        for method, stats in self.method_stats.items():
            success_rate = (stats["success_count"] / stats["count"]) * 100 if stats["count"] > 0 else 0
            method_analysis[method] = {
                "question_count": stats["count"],
                "success_count": stats["success_count"],
                "success_rate": round(success_rate, 1),
                "avg_processing_time": round(stats["avg_time"], 3),
                "total_processing_time": round(stats["total_time"], 3)
            }
        
        return method_analysis
    
    def _calculate_performance_metrics(self) -> Dict:
        """성능 지표 계산"""
        if not self.processing_times:
            return {}
            
        processing_times = sorted(self.processing_times)
        
        return {
            "min_processing_time": round(min(processing_times), 3),
            "max_processing_time": round(max(processing_times), 3),
            "median_processing_time": round(processing_times[len(processing_times) // 2], 3),
            "p95_processing_time": round(processing_times[int(len(processing_times) * 0.95)], 3),
            "p99_processing_time": round(processing_times[int(len(processing_times) * 0.99)], 3),
            "std_deviation": round(self._calculate_std_deviation(processing_times), 3),
            "efficiency_score": self._calculate_efficiency_score()
        }
    
    def _analyze_learning_data(self, learning_data: Dict) -> Dict:
        """학습 데이터 분석"""
        total_successful = learning_data.get("successful_answers", 0)
        total_failed = learning_data.get("failed_answers", 0)
        total_patterns = learning_data.get("question_patterns", 0)
        
        learning_rate = (total_successful / (total_successful + total_failed)) * 100 if (total_successful + total_failed) > 0 else 0
        
        return {
            "successful_answers": total_successful,
            "failed_answers": total_failed,
            "pattern_records": total_patterns,
            "learning_rate": round(learning_rate, 1),
            "knowledge_accumulation": total_successful + total_patterns,
            "error_recovery_data": total_failed
        }
    
    def _analyze_system_metrics(self) -> Dict:
        """시스템 지표 분석"""
        if not self.memory_snapshots:
            return {}
            
        memory_usages = [snap["used_percent"] for snap in self.memory_snapshots]
        
        return {
            "avg_memory_usage": round(sum(memory_usages) / len(memory_usages), 1),
            "peak_memory_usage": round(max(memory_usages), 1),
            "memory_snapshots": len(self.memory_snapshots),
            "system_stability": "정상" if max(memory_usages) < 85 else "주의필요"
        }
    
    def _calculate_std_deviation(self, values: List[float]) -> float:
        """표준편차 계산"""
        if len(values) < 2:
            return 0
            
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _calculate_efficiency_score(self) -> float:
        """처리 효율 점수 계산"""
        if not self.processing_times:
            return 0
            
        avg_time = sum(self.processing_times) / len(self.processing_times)
        consistency = 1 - (self._calculate_std_deviation(self.processing_times) / avg_time) if avg_time > 0 else 0
        speed_score = max(0, 1 - (avg_time / 10))  # 10초를 기준으로 속도 점수
        
        return round((consistency * 0.6 + speed_score * 0.4) * 100, 1)
    
    def _write_detailed_log(self, stats: Dict):
        """상세 로그 작성"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 실행 요약
                exec_summary = stats["execution_summary"]
                f.write(f"[{timestamp}] === 실행 통계 ===\n")
                f.write(f"[{timestamp}] 총 처리 시간: {exec_summary['total_time_minutes']}분 ({exec_summary['total_time_seconds']}초)\n")
                f.write(f"[{timestamp}] 처리 문항 수: {exec_summary['total_questions']}개\n")
                f.write(f"[{timestamp}] 평균 문항 처리 시간: {exec_summary['avg_processing_time']}초\n")
                f.write(f"[{timestamp}] 분당 처리량: {exec_summary['questions_per_minute']}문항/분\n")
                
                # 도메인 분석
                if stats["domain_analysis"]:
                    f.write(f"[{timestamp}] === 도메인별 분석 ===\n")
                    for domain, analysis in stats["domain_analysis"].items():
                        f.write(f"[{timestamp}] {domain}: {analysis['question_count']}문항 ({analysis['percentage']}%), "
                               f"평균 {analysis['avg_processing_time']}초\n")
                
                # 처리 방법 분석
                if stats["method_analysis"]:
                    f.write(f"[{timestamp}] === 처리 방법별 분석 ===\n")
                    for method, analysis in stats["method_analysis"].items():
                        f.write(f"[{timestamp}] {method}: {analysis['question_count']}문항, "
                               f"성공률 {analysis['success_rate']}%, 평균 {analysis['avg_processing_time']}초\n")
                
                # 성능 지표
                if stats["performance_metrics"]:
                    perf = stats["performance_metrics"]
                    f.write(f"[{timestamp}] === 성능 지표 ===\n")
                    f.write(f"[{timestamp}] 처리 시간 범위: {perf['min_processing_time']}초 ~ {perf['max_processing_time']}초\n")
                    f.write(f"[{timestamp}] 중간값: {perf['median_processing_time']}초, P95: {perf['p95_processing_time']}초\n")
                    f.write(f"[{timestamp}] 표준편차: {perf['std_deviation']}초, 효율점수: {perf['efficiency_score']}점\n")
                
                # 학습 지표
                if stats["learning_metrics"]:
                    learning = stats["learning_metrics"]
                    f.write(f"[{timestamp}] === 학습 데이터 분석 ===\n")
                    f.write(f"[{timestamp}] 성공 답변: {learning['successful_answers']}개\n")
                    f.write(f"[{timestamp}] 실패 답변: {learning['failed_answers']}개\n")
                    f.write(f"[{timestamp}] 패턴 기록: {learning['pattern_records']}개\n")
                    f.write(f"[{timestamp}] 학습 성공률: {learning['learning_rate']}%\n")
                    f.write(f"[{timestamp}] 누적 지식량: {learning['knowledge_accumulation']}개\n")
                
                # 시스템 지표
                if stats["system_metrics"]:
                    system = stats["system_metrics"]
                    f.write(f"[{timestamp}] === 시스템 지표 ===\n")
                    f.write(f"[{timestamp}] 평균 메모리 사용률: {system['avg_memory_usage']}%\n")
                    f.write(f"[{timestamp}] 최대 메모리 사용률: {system['peak_memory_usage']}%\n")
                    f.write(f"[{timestamp}] 시스템 상태: {system['system_stability']}\n")
                
                # 오류 분석
                if stats["error_analysis"]:
                    f.write(f"[{timestamp}] === 오류 분석 ===\n")
                    for error_type, count in stats["error_analysis"].items():
                        f.write(f"[{timestamp}] {error_type}: {count}회\n")
                
                # 템플릿 사용 분석
                if stats["template_analysis"]:
                    f.write(f"[{timestamp}] === 템플릿 사용 분석 ===\n")
                    for template_type, count in stats["template_analysis"].items():
                        f.write(f"[{timestamp}] {template_type}: {count}회\n")
                
                f.write(f"[{timestamp}] === 세션 종료 ===\n\n")
                
        except Exception as e:
            pass