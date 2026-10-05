const form = document.querySelector("form");

form.addEventListener("submit", function(event) {
    event.preventDefault();

    // Numerical features
    const hoursStudied = Number(document.getElementById("Hours_Studied").value);
    const attendance = Number(document.getElementById("Attendance").value);
    const sleepHours = Number(document.getElementById("Sleep_Hours").value);
    const previousScores = Number(document.getElementById("Previous_Scores").value);
    const tutoringSessions = Number(document.getElementById("Tutoring_Sessions").value);
    const physicalActivity = Number(document.getElementById("Physical_Activity").value);

    // Categorical features
    const parentalInvolvement = document.getElementById("Parental_Involvement").value;
    const accessToResources = document.getElementById("Access_to_Resources").value;
    const extracurricularActivities = document.getElementById("Extracurricular_Activities").value;
    const motivationLevel = document.getElementById("Motivation_Level").value;
    const internetAccess = document.getElementById("Internet_Access").value;
    const familyIncome = document.getElementById("Family_Income").value;
    const teacherQuality = document.getElementById("Teacher_Quality").value;
    const schoolType = document.getElementById("School_Type").value;
    const peerInfluence = document.getElementById("Peer_Influence").value;
    const learningDisabilities = document.getElementById("Learning_Disabilities").value;
    const parentalEducationLevel = document.getElementById("Parental_Education_Level").value;
    const distanceFromHome = document.getElementById("Distance_from_Home").value;
    const gender = document.getElementById("Gender").value;

    const studentData = {
    Hours_Studied: hoursStudied,
    Attendance: attendance,
    Sleep_Hours: sleepHours,
    Previous_Scores: previousScores,
    Tutoring_Sessions: tutoringSessions,
    Physical_Activity: physicalActivity,

    Parental_Involvement: parentalInvolvement,
    Access_to_Resources: accessToResources,
    Extracurricular_Activities: extracurricularActivities,
    Motivation_Level: motivationLevel,
    Internet_Access: internetAccess,
    Family_Income: familyIncome,
    Teacher_Quality: teacherQuality,
    School_Type: schoolType,
    Peer_Influence: peerInfluence,
    Learning_Disabilities: learningDisabilities,
    Parental_Education_Level: parentalEducationLevel,
    Distance_from_Home: distanceFromHome,
    Gender: gender
};

const results = document.getElementById("results");

console.log(studentData);

fetch("http://127.0.0.1:8000/predict", {
    method: "POST",

    headers: {
        "Content-Type": "application/json"
    },

    body: JSON.stringify(studentData)
})
.then(response => response.json())
.then(data => {
    console.log("Prediction received:");
    console.log(data);

    results.innerHTML = `
    <h2>Prediction Results</h2>

    <p>Linear Regression: ${data["Linear Regression"]}</p>
    <p>Ridge Regression: ${data["Ridge Regression"]}</p>
    <p>Lasso Regression: ${data["Lasso Regression"]}</p>
`;
});
});

