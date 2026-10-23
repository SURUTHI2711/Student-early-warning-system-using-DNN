/**
 * Main Client-Side Script
 * Project: At-Risk Student Early Warning System Using Deep Neural Networks
 */

document.addEventListener("DOMContentLoaded", function () {
  // 1. Auto-dismiss Flash Alerts after 5 seconds
  const flashAlerts = document.querySelectorAll(".alert-box");
  flashAlerts.forEach(function (alert) {
    setTimeout(function () {
      alert.style.transition = "opacity 0.6s ease, transform 0.6s ease";
      alert.style.opacity = "0";
      alert.style.transform = "translateY(-8px)";
      setTimeout(() => alert.remove(), 600);
    }, 5500);
  });

  // 2. Demo Credential Quick-Fill on Login Page
  const btnFillAdmin = document.getElementById("btn-fill-admin");
  const btnFillStudent = document.getElementById("btn-fill-student");
  const usernameInput = document.getElementById("username");
  const passwordInput = document.getElementById("password");

  if (btnFillAdmin && usernameInput && passwordInput) {
    btnFillAdmin.addEventListener("click", function () {
      usernameInput.value = "admin";
      passwordInput.value = "admin123";
      usernameInput.focus();
    });
  }

  if (btnFillStudent && usernameInput && passwordInput) {
    btnFillStudent.addEventListener("click", function () {
      usernameInput.value = "student1";
      passwordInput.value = "student123";
      usernameInput.focus();
    });
  }

  // 3. Dynamic Student Selector on Prediction Page
  const studentSelector = document.getElementById("existing-student-selector");
  if (studentSelector) {
    studentSelector.addEventListener("change", function () {
      const selectedOpt = this.options[this.selectedIndex];
      if (!selectedOpt.value) return;

      const sid = selectedOpt.getAttribute("data-id");
      const name = selectedOpt.getAttribute("data-name");
      const dept = selectedOpt.getAttribute("data-dept");
      const year = selectedOpt.getAttribute("data-year");
      const sem = selectedOpt.getAttribute("data-sem");

      if (document.getElementById("student_id")) document.getElementById("student_id").value = sid || "";
      if (document.getElementById("student_name")) document.getElementById("student_name").value = name || "";
      if (document.getElementById("department")) document.getElementById("department").value = dept || "";
      if (document.getElementById("year")) document.getElementById("year").value = year || 3;
      if (document.getElementById("semester")) document.getElementById("semester").value = sem || 6;

      // Also prefill academic numbers if available
      const att = selectedOpt.getAttribute("data-att");
      const marks = selectedOpt.getAttribute("data-marks");
      const gpa = selectedOpt.getAttribute("data-gpa");
      const backlogs = selectedOpt.getAttribute("data-backlogs");

      if (att && document.getElementById("attendance_percentage")) document.getElementById("attendance_percentage").value = att;
      if (marks && document.getElementById("internal_marks")) document.getElementById("internal_marks").value = marks;
      if (gpa && document.getElementById("previous_gpa")) document.getElementById("previous_gpa").value = gpa;
      if (backlogs && document.getElementById("backlogs")) document.getElementById("backlogs").value = backlogs;
    });
  }

  // 4. Form Submission Loading Feedback
  const predictForm = document.getElementById("prediction-form");
  const btnPredict = document.getElementById("btn-predict-submit");
  if (predictForm && btnPredict) {
    predictForm.addEventListener("submit", function () {
      btnPredict.disabled = true;
      btnPredict.innerHTML = `<span class="spinner" style="display:inline-block;width:14px;height:14px;border:2px solid #fff;border-top-color:transparent;border-radius:50%;animation:spin 0.8s linear infinite;margin-right:6px;"></span> Evaluating DNN...`;
    });
  }
});
