
import 'firebase/compat/auth'

import { initializeApp, getApps } from "firebase/app";

const firebaseConfig = {
  apiKey: "AIzaSyDXDGGUots5JHk39kfGGV5ueRd09Ot3f50",
  authDomain: "yceffort.firebaseapp.com",
  databaseURL: "https://yceffort.firebaseio.com",
  projectId: "yceffort",
  storageBucket: "yceffort.appspot.com",
  messagingSenderId: "754165146494",
  appId: "1:754165146494:web:41d36183a76fb998f4892f",
  measurementId: "G-PEKGCL9BKE"
} as const

const firebaseApp = getApps().length === 0  ? initializeApp(firebaseConfig) : getApps()[0]


export default firebaseApp
