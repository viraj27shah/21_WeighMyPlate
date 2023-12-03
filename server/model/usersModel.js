const mongoose = require('mongoose');

const Schema =  mongoose.Schema;

const usersSchema = new Schema({
    name:{
        type : String,
        // required : true
    },
    email:{
        type : String,
        unique:true,
        // required : true
    },
    password:{
        type : String,
        // required : true
    },
    isActive: {
        type : Boolean,
        default : true
        // required : true
    },
    isDeleted: {
        type : Boolean,
        default : false
        // required : true
    }
}, {timestamps:true});

module.exports = mongoose.model('user',usersSchema);