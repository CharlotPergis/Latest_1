// Full History Page - Enhanced with CSV Download
(function() {
  // DOM Elements
  const fullLogBody = document.getElementById("full-log-body");
  const totalCountSpan = document.getElementById("total-count");
  const filteredCountSpan = document.getElementById("filtered-count");
  const startDateInput = document.getElementById("startDate");
  const endDateInput = document.getElementById("endDate");
  const startTimeInput = document.getElementById("startTime");
  const endTimeInput = document.getElementById("endTime");
  const statusFilter = document.getElementById("statusFilter");
  const searchInput = document.getElementById("searchInput");
  const dateRangePreset = document.getElementById("dateRangePreset");
  const applyFiltersBtn = document.getElementById("applyFiltersBtn");
  const clearFiltersBtn = document.getElementById("clearFiltersBtn");
  const downloadCsvBtn = document.getElementById("downloadCsvBtn");
  const quick1h = document.getElementById("quick1h");
  const quick24h = document.getElementById("quick24h");
  const quick7d = document.getElementById("quick7d");

  // Data stores
  let allLogs = [];
  let currentFilteredLogs = [];

  // Filter state
  let activeFilters = {
    dateRange: { start: null, end: null },
    status: "all",
    search: "",
    quickFilter: null
  };

  // Helper: Parse timestamp
  function parseTimestamp(timestamp) {
    if (!timestamp) return null;
    
    if (typeof timestamp === 'string' && timestamp.includes(' ') && timestamp.includes('-')) {
      const [datePart, timePart] = timestamp.split(' ');
      if (datePart && timePart) {
        const isoString = `${datePart}T${timePart}`;
        const date = new Date(isoString);
        if (!isNaN(date.getTime())) return date;
      }
    }
    
    const date = new Date(timestamp);
    if (!isNaN(date.getTime())) return date;
    
    return null;
  }

  // Format date for display
  function formatDisplayDate(dateObj) {
    if (!dateObj || isNaN(dateObj.getTime())) return "Invalid Date";
    const day = String(dateObj.getDate()).padStart(2, '0');
    const month = String(dateObj.getMonth() + 1).padStart(2, '0');
    const year = dateObj.getFullYear();
    const hours = String(dateObj.getHours()).padStart(2, '0');
    const minutes = String(dateObj.getMinutes()).padStart(2, '0');
    const seconds = String(dateObj.getSeconds()).padStart(2, '0');
    return `${day}/${month}/${year} ${hours}:${minutes}:${seconds}`;
  }

  // Create Date object from inputs
  function createDateFromInputs(dateValue, timeValue) {
    if (!dateValue) return null;
    const timeStr = timeValue || "00:00";
    const dateTimeStr = `${dateValue}T${timeStr}:00`;
    const date = new Date(dateTimeStr);
    return isNaN(date.getTime()) ? null : date;
  }

  // Apply all active filters
  function applyAllFilters() {
    let filtered = [...allLogs];
    
    // 1. Date range filter
    if (activeFilters.dateRange.start) {
      filtered = filtered.filter(log => log.timestamp >= activeFilters.dateRange.start);
    }
    if (activeFilters.dateRange.end) {
      filtered = filtered.filter(log => log.timestamp <= activeFilters.dateRange.end);
    }
    
    // 2. Status filter
    if (activeFilters.status !== "all") {
      filtered = filtered.filter(log => {
        if (activeFilters.status === "normal") return log.breakerState === "Normal";
        if (activeFilters.status === "warning") return log.breakerState === "Potential Overload";
        if (activeFilters.status === "overload") return log.breakerState === "Overload";
        if (activeFilters.status === "danger") return log.breakerState === "Overheating";
        return true;
      });
    }
    
    // 3. Search filter
    if (activeFilters.search.trim()) {
      const searchTerm = activeFilters.search.toLowerCase().trim();
      filtered = filtered.filter(log => {
        return log.breakerState.toLowerCase().includes(searchTerm) ||
               log.systemStatus.toLowerCase().includes(searchTerm) ||
               log.temperature.toString().includes(searchTerm) ||
               log.current.toString().includes(searchTerm);
      });
    }
    
    currentFilteredLogs = filtered;
    renderTable();
    updateStats();
  }

  // Update statistics display
  function updateStats() {
    if (totalCountSpan) totalCountSpan.textContent = allLogs.length;
    if (filteredCountSpan) filteredCountSpan.textContent = currentFilteredLogs.length;
    const filterStatsSpan = document.getElementById("filterStats");
    if (filterStatsSpan) {
      filterStatsSpan.textContent = `${currentFilteredLogs.length} logs`;
    }
  }

  // Set date range preset
  function setDateRangePreset(preset) {
    const now = new Date();
    let start = null;
    let end = null;
    
    switch(preset) {
      case "today":
        start = new Date(now);
        start.setHours(0, 0, 0, 0);
        end = new Date(now);
        end.setHours(23, 59, 59, 999);
        break;
      case "yesterday":
        start = new Date(now);
        start.setDate(start.getDate() - 1);
        start.setHours(0, 0, 0, 0);
        end = new Date(now);
        end.setDate(end.getDate() - 1);
        end.setHours(23, 59, 59, 999);
        break;
      case "week":
        start = new Date(now);
        start.setDate(start.getDate() - 7);
        start.setHours(0, 0, 0, 0);
        break;
      case "month":
        start = new Date(now);
        start.setMonth(start.getMonth() - 1);
        start.setHours(0, 0, 0, 0);
        break;
      case "all":
        start = null;
        end = null;
        break;
      default:
        return;
    }
    
    activeFilters.dateRange.start = start;
    activeFilters.dateRange.end = end;
    
    // Update input fields
    if (startDateInput && start) {
      startDateInput.value = start.toISOString().split('T')[0];
      if (startTimeInput) startTimeInput.value = "00:00";
    }
    if (endDateInput && end) {
      endDateInput.value = end.toISOString().split('T')[0];
      if (endTimeInput) endTimeInput.value = "23:59";
    }
    
    applyAllFilters();
  }

  // Load data from localStorage
  function loadData() {
    try {
      const stored = localStorage.getItem("breakerFullHistory");
      let rawData = stored ? JSON.parse(stored) : [];
      
      if (rawData.length === 0) {
        rawData = generateSampleData();
        localStorage.setItem("breakerFullHistory", JSON.stringify(rawData));
      }
      
      allLogs = rawData
        .map(item => {
          const timestampObj = parseTimestamp(item.timestamp);
          if (!timestampObj) return null;
          
          return {
            originalTimestamp: item.timestamp,
            timestamp: timestampObj,
            temperature: typeof item.temperature === 'number' ? item.temperature : parseFloat(item.temperature) || 0,
            current: typeof item.current === 'number' ? item.current : parseFloat(item.current) || 0,
            breakerState: item.breakerState || "Unknown",
            systemStatus: item.systemStatus || "No status"
          };
        })
        .filter(item => item !== null);
      
      allLogs.sort((a, b) => b.timestamp - a.timestamp);
      
      clearAllFilters();
      
    } catch (error) {
      console.error("Error loading data:", error);
      allLogs = [];
      currentFilteredLogs = [];
      renderTable();
    }
  }
  
  // Generate sample data
  function generateSampleData() {
    const sampleData = [];
    const now = new Date();
    const states = [
      { name: "Normal", tempRange: [20, 40], currentRange: [5, 15], status: "✅ System operational - Normal" },
      { name: "Potential Overload", tempRange: [45, 65], currentRange: [15, 20], status: "⚡ ALERT - Load approaching limits" },
      { name: "Overload", tempRange: [65, 80], currentRange: [20, 25], status: "⚠️ WARNING - High load detected" },
      { name: "Overheating", tempRange: [80, 95], currentRange: [22, 28], status: "🔥 CRITICAL - Temperature exceeded" }
    ];
    
    for (let i = 0; i < 150; i++) {
      const hoursAgo = i * 2;
      const date = new Date(now.getTime() - hoursAgo * 3600000);
      
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const day = String(date.getDate()).padStart(2, '0');
      const hours = String(date.getHours()).padStart(2, '0');
      const minutes = String(date.getMinutes()).padStart(2, '0');
      const seconds = String(date.getSeconds()).padStart(2, '0');
      const timestamp = `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
      
      const stateIndex = Math.random() < 0.7 ? 0 : Math.floor(Math.random() * 3) + 1;
      const state = states[stateIndex];
      
      const temperature = state.tempRange[0] + Math.random() * (state.tempRange[1] - state.tempRange[0]);
      const current = state.currentRange[0] + Math.random() * (state.currentRange[1] - state.currentRange[0]);
      
      sampleData.push({
        timestamp: timestamp,
        temperature: Math.round(temperature * 10) / 10,
        current: Math.round(current * 10) / 10,
        breakerState: state.name,
        systemStatus: state.status
      });
    }
    
    return sampleData;
  }
  
  // Render table
  function renderTable() {
    if (!fullLogBody) return;
    
    fullLogBody.innerHTML = "";
    
    if (!currentFilteredLogs || currentFilteredLogs.length === 0) {
      const emptyRow = document.createElement("tr");
      emptyRow.className = "empty-row";
      emptyRow.innerHTML = '<td colspan="5" style="text-align:center; padding: 48px;">📭 No logs match the selected filters. Try adjusting your filters.<\/td>';
      fullLogBody.appendChild(emptyRow);
      updateStats();
      return;
    }
    
    const fragment = document.createDocumentFragment();
    
    currentFilteredLogs.forEach(log => {
      let stateClass = "state-normal";
      if (log.breakerState === "Overload") stateClass = "state-overload";
      else if (log.breakerState === "Potential Overload") stateClass = "state-warning";
      else if (log.breakerState === "Overheating") stateClass = "state-danger";
      
      const row = document.createElement("tr");
      const displayDate = formatDisplayDate(log.timestamp);
      
      row.innerHTML = `
        <td style="font-weight: 500;">${displayDate}<\/td>
        <td>${log.temperature.toFixed(1)} °C<\/td>
        <td>${log.current.toFixed(1)} A<\/td>
        <td class="${stateClass}">${log.breakerState}<\/td>
        <td>${log.systemStatus || '—'}<\/td>
      `;
      fragment.appendChild(row);
    });
    
    fullLogBody.appendChild(fragment);
    updateStats();
  }
  
  // Clear all filters
  function clearAllFilters() {
    activeFilters = {
      dateRange: { start: null, end: null },
      status: "all",
      search: "",
      quickFilter: null
    };
    
    if (startDateInput) startDateInput.value = "";
    if (endDateInput) endDateInput.value = "";
    if (startTimeInput) startTimeInput.value = "";
    if (endTimeInput) endTimeInput.value = "";
    if (statusFilter) statusFilter.value = "all";
    if (searchInput) searchInput.value = "";
    if (dateRangePreset) dateRangePreset.value = "custom";
    
    currentFilteredLogs = [...allLogs];
    renderTable();
  }
  
  // Show toast notification
  function showToast(message, isError = false) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.style.backgroundColor = isError ? '#ef4444' : '#22c55e';
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
      toast.remove();
    }, 3000);
  }
  
  // Download CSV
  function downloadCSV() {
    if (!currentFilteredLogs || currentFilteredLogs.length === 0) {
      showToast("⚠️ No logs available to download. Please adjust your filters.", true);
      return;
    }
    
    try {
      const headers = ["Date Time", "Temperature (°C)", "Current (A)", "Breaker State", "Status"];
      const csvRows = [headers];
      
      currentFilteredLogs.forEach(log => {
        csvRows.push([
          formatDisplayDate(log.timestamp),
          log.temperature.toFixed(1),
          log.current.toFixed(1),
          log.breakerState,
          log.systemStatus || "N/A"
        ]);
      });
      
      const csvContent = csvRows.map(row => 
        row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(",")
      ).join("\n");
      
      const blob = new Blob(["\uFEFF" + csvContent], { type: "text/csv;charset=utf-8;" });
      const link = document.createElement("a");
      const url = URL.createObjectURL(blob);
      const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
      
      link.href = url;
      link.download = `breaker_logs_${timestamp}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      showToast(`✅ Downloaded ${currentFilteredLogs.length} logs successfully!`);
      
    } catch (error) {
      console.error("CSV Download error:", error);
      showToast("❌ Error downloading CSV. Please try again.", true);
    }
  }
  
  // Quick filter by hours
  function filterByHours(hours) {
    const now = new Date();
    const cutoffTime = new Date(now.getTime() - hours * 3600000);
    
    activeFilters.dateRange.start = cutoffTime;
    activeFilters.dateRange.end = null;
    activeFilters.quickFilter = hours;
    
    if (statusFilter) statusFilter.value = "all";
    if (searchInput) searchInput.value = "";
    activeFilters.status = "all";
    activeFilters.search = "";
    
    if (startDateInput) startDateInput.value = "";
    if (endDateInput) endDateInput.value = "";
    if (startTimeInput) startTimeInput.value = "";
    if (endTimeInput) endTimeInput.value = "";
    if (dateRangePreset) dateRangePreset.value = "custom";
    
    applyAllFilters();
    showToast(`📊 Showing logs from last ${hours} hour${hours > 1 ? 's' : ''}`);
  }
  
  // Auto-refresh
  function startAutoRefresh() {
    setInterval(() => {
      try {
        const stored = localStorage.getItem("breakerFullHistory");
        if (stored) {
          const newData = JSON.parse(stored);
          if (newData.length !== allLogs.length) {
            loadData();
          }
        }
      } catch (e) {
        console.error("Auto-refresh error:", e);
      }
    }, 3000);
  }
  
  // Event Listeners
  if (applyFiltersBtn) applyFiltersBtn.addEventListener("click", applyAllFilters);
  if (clearFiltersBtn) clearFiltersBtn.addEventListener("click", clearAllFilters);
  if (downloadCsvBtn) downloadCsvBtn.addEventListener("click", downloadCSV);
  if (quick1h) quick1h.addEventListener("click", () => filterByHours(1));
  if (quick24h) quick24h.addEventListener("click", () => filterByHours(24));
  if (quick7d) quick7d.addEventListener("click", () => filterByHours(168));
  
  if (statusFilter) {
    statusFilter.addEventListener("change", () => {
      activeFilters.status = statusFilter.value;
      applyAllFilters();
    });
  }
  
  if (searchInput) {
    let searchTimeout;
    searchInput.addEventListener("input", (e) => {
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(() => {
        activeFilters.search = e.target.value;
        applyAllFilters();
      }, 300);
    });
  }
  
  if (dateRangePreset) {
    dateRangePreset.addEventListener("change", (e) => {
      if (e.target.value !== "custom") {
        setDateRangePreset(e.target.value);
      }
    });
  }
  
  if (startDateInput) {
    startDateInput.addEventListener("change", () => {
      activeFilters.dateRange.start = createDateFromInputs(startDateInput.value, startTimeInput?.value);
      if (dateRangePreset) dateRangePreset.value = "custom";
      applyAllFilters();
    });
  }
  
  if (endDateInput) {
    endDateInput.addEventListener("change", () => {
      activeFilters.dateRange.end = createDateFromInputs(endDateInput.value, endTimeInput?.value);
      if (dateRangePreset) dateRangePreset.value = "custom";
      applyAllFilters();
    });
  }
  
  if (startTimeInput) {
    startTimeInput.addEventListener("change", () => {
      activeFilters.dateRange.start = createDateFromInputs(startDateInput?.value, startTimeInput.value);
      if (dateRangePreset) dateRangePreset.value = "custom";
      applyAllFilters();
    });
  }
  
  if (endTimeInput) {
    endTimeInput.addEventListener("change", () => {
      activeFilters.dateRange.end = createDateFromInputs(endDateInput?.value, endTimeInput.value);
      if (dateRangePreset) dateRangePreset.value = "custom";
      applyAllFilters();
    });
  }
  
  window.addEventListener("storage", (event) => {
    if (event.key === "breakerFullHistory") {
      loadData();
    }
  });
  
  // Initialize
  loadData();
  startAutoRefresh();
})();