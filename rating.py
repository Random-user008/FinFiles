import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
  page_title="GitHub Project Recommendation System",
  page_icon="GitHub-icon.png",
)
components.html(""" <head>
    <style>
    @import url('https://fonts.googleapis.com/css?family=Poppins:400,500,600,700&display=swap');
*{
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  font-family: 'Poppins', sans-serif;
}
html,body{
  display: grid;
  height: 100%;
  place-items: center;
  text-align: center;
  background: white;
}
.container{
  position: relative;
  width: 400px;
  background: white;
  padding: 20px 30px;
  border: 1px solid #444;
  border-radius: 5px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
}
.container .post{
  display: none;
}
.container .text{
  font-size: 25px;
  color: #666;
  font-weight: 500;
}
.container .edit{
  position: absolute;
  right: 10px;
  top: 5px;
  font-size: 16px;
  color: #666;
  font-weight: 500;
  cursor: pointer;
}
.container .edit:hover{
  text-decoration: underline;
}
.container .star-widget input{
  display: none;
}
.star-widget label{
  font-size: 40px;
  color: #444;
  padding: 10px;
  float: right;
  transition: all 0.2s ease;
}
input:not(:checked) ~ label:hover,
input:not(:checked) ~ label:hover ~ label{
  color: rgb(125, 0, 241);
}
input:checked ~ label{
  color: rgb(125, 0, 241);
}
input#rate-5:checked ~ label{
  color: rgb(125, 0, 241);
  text-shadow: 0 0 20px rgb(255, 238, 0);
}
#rate-1:checked ~ form header:before{
  content: "Very Dissatisfied ";
}
#rate-2:checked ~ form header:before{
  content: "Dissatisfied";
}
#rate-3:checked ~ form header:before{
  content: "Neutral ";
}
#rate-4:checked ~ form header:before{
  content: "Satisfied";
}
#rate-5:checked ~ form header:before{
  content: "Very Satisfied";
}
.container form{
  display: none;
}
input:checked ~ form{
  display: block;
}
form header{
  width: 100%;
  font-size: 25px;
  color: rgb(125, 0, 241);
  font-weight: 500;
  margin: 5px 0 20px 0;
  text-align: center;
  transition: all 0.2s ease;
}
form .textarea{
  height: 100px;
  width: 100%;
  overflow: hidden;
}
form .textarea textarea{
  height: 100%;
  width: 100%;
  outline: none;
  color: black;
  border: 1px solid #333;
  background: white;
  padding: 10px;
  font-size: 17px;
  resize: none;
}
.textarea textarea:focus{
  border-color: #444;
}
form .btn{
  height: 45px;
  width: 100%;
  margin: 15px 0;
}
form .btn button{
  height: 100%;
  width: 100%;
  border: 1px solid #444;
  outline: none;
  background: rgb(125, 0, 241);
  color: white;
  font-size: 17px;
  font-weight: 500;
  text-transform: uppercase;
  cursor: pointer;
  transition: all 0.3s ease;
}
form .btn button:hover{
  background: #1b1b1b;
}
    </style>
  <head>
   
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"/>
  </head>
  <body>
    <div class="container">
      <div class="post">
        <div class="text">Thanks for rating us!</div>
        <div class="edit">BACK</div>
      </div>
      <div class="star-widget">
        <input type="radio" name="rate" id="rate-5" value="10%">
        <label for="rate-5" class="fas fa-star"></label>
        <input type="radio" name="rate" id="rate-4" value="80">
        <label for="rate-4" class="fas fa-star"></label>
        <input type="radio" name="rate" id="rate-3" value="60">
        <label for="rate-3" class="fas fa-star"></label>
        <input type="radio" name="rate" id="rate-2" value="40">
        <label for="rate-2" class="fas fa-star"></label>
        <input type="radio" name="rate" id="rate-1" value="20">
        <label for="rate-1" class="fas fa-star"></label>
        <form action="" method="POST">
          <header></header>
          <div class="textarea">
            <textarea cols="30" placeholder="Describe your experience.."></textarea>
          </div>
          <div class="btn">
            <button type="submit" onclick="myFunction()">Post</button>
          </div>
          <div id="result"></div>
        </form>
        
      </div>
    </div>
    <script>
      function myFunction() {
    var ele = document.getElementsByName('rate');
          
          for(i = 0; i < ele.length; i++) {
              if(ele[i].checked)
                const myVariable =+ele[i].value;//stores value of user feedback
                const MongoClient = require('mongodb').MongoClient;

// Replace with your MongoDB Atlas connection string
const uri = "mongosh "mongodb+srv://finalyearproject.saacj6j.mongodb.net/Reporecommender" --apiVersion 1 --username Rishabh175";
const client = new MongoClient(uri, { useNewUrlParser: true, useUnifiedTopology: true });

// Connect to the MongoDB Atlas database
client.connect(err => {
  const collection = client.db("Reporecommender").collection("userfeedback");
  
  // Insert the variable value in the database
  const document = { value: myVariable };
  
  collection.insertOne(document, (err, result) => {
    if (err) {
      console.log(err);
    } else {
      console.log(`Inserted variable value: ${myVariable}`);
    }
    
    // Disconnect from the database
    client.close();
  });
});

          }
  
   } 
      const btn = document.querySelector("button");
      const post = document.querySelector(".post");
      const widget = document.querySelector(".star-widget");
      const editBtn = document.querySelector(".edit");
      btn.onclick = ()=>{
        widget.style.display = "none";
        post.style.display = "block";
        editBtn.onclick = ()=>{
          widget.style.display = "block";
          post.style.display = "none";
        }
        return false;
      }
    </script>

  </body>    """,
                height=400,
                width=700)
