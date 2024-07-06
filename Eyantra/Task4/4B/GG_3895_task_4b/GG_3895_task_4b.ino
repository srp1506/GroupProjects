// Motor pin connections
#define in1 16
#define in2 4
#define in3 2
#define in4 15
#define led 13
#define buzzer 12

// IR sensor pin connections
#define olir 17  // Tx2
#define orir 5  // D5
#define ilir 19  // D19
#define irir 18  // D18
int nodescrossed = 0;

void led_on() {

  digitalWrite(led, HIGH);
}

void led_off() {

  digitalWrite(led, LOW);
}

void buzzer_on() {

  digitalWrite(buzzer, HIGH);
}

void buzzer_off() {

  digitalWrite(buzzer, LOW);
}

void moveForward() {
  
  digitalWrite(in3, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in1, HIGH);
  digitalWrite(in4, HIGH);
}

// 2r 3l 4r 5r 7r 9l

void right() {
  digitalWrite(in4, HIGH);
 // digitalWrite(in1, LOW);
}

void left() {
 // digitalWrite(in4, LOW);
  digitalWrite(in1, HIGH);
}

void Pass() {
  while (digitalRead(irir) == HIGH && digitalRead(ilir) == HIGH){
    //digitalWrite(led, HIGH);
    digitalWrite(in1, HIGH);
  digitalWrite(in4, HIGH);
  delay(20);
    digitalWrite(in1, LOW);
  digitalWrite(in4, LOW);
 // digitalWrite(led, LOW);
  }
}

void Right() {
      digitalWrite(in4, HIGH);
        digitalWrite(in1, LOW);
      delay(800);
   while (digitalRead(irir) == HIGH && digitalRead(ilir) == HIGH ){
  //  digitalWrite(led, HIGH);
    digitalWrite(in4, HIGH);
  digitalWrite(in1, LOW);
   delay(20);
    digitalWrite(in1, LOW);
  digitalWrite(in4, LOW);
  }
}

void Left(){
      digitalWrite(in1, HIGH);
        digitalWrite(in4, LOW);
      delay(800);
   while (digitalRead(irir) == HIGH && digitalRead(ilir) == HIGH ){
  //  digitalWrite(led, HIGH);
    digitalWrite(in1, HIGH);
  digitalWrite(in4, LOW);
   delay(20);
    digitalWrite(in1, LOW);
  digitalWrite(in4, LOW);
  }
}

void nr() {
  digitalWrite(in4, LOW);
 // digitalWrite(in1, LOW);
}

void nl() {
 // digitalWrite(in4, LOW);
  digitalWrite(in1, LOW);
}


void Halt() {

  digitalWrite(in3, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in4, LOW);
  digitalWrite(in1, LOW);
}

void setup() {
  // Motor pins as output
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  pinMode(led, OUTPUT);
  pinMode(buzzer, OUTPUT);

  // IR sensor pins as input
  pinMode(olir, INPUT);
  pinMode(orir, INPUT);
  pinMode(ilir, INPUT);
  pinMode(irir, INPUT);
  
  // Keeping all motors off initially
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);

  led_on();
  buzzer_on();
  delay(1000);
  led_off();
  buzzer_off();

   int orirv = digitalRead(orir);
   int irirv = digitalRead(irir);
   int ilirv = digitalRead(ilir);
   int olirv = digitalRead(olir);
}

void loop() {
int ter = -1;
int ret = 0;
int las = 0;
int sal = 0;

  while(true){
  // delay(3000);
   int orirv = digitalRead(orir);
   int irirv = digitalRead(irir);
   int ilirv = digitalRead(ilir);
   int olirv = digitalRead(olir);
   int dune = 1;

 // Halt();
  if (ilirv == LOW && irirv == LOW ) {
    nr();
    //delay(0.1);
    if (orirv == LOW ) right();
    else nr();
    if (olirv == LOW ) left();
    else nl();
  
    }

  else {

    if ( irirv == LOW ){ left(); dune = 0;}
    else nl();
    if ( ilirv == LOW ){right(); dune = 0;}
    else nr();

    if(dune == 1){

    moveForward();
    delay(50);

    if(irirv == HIGH && ilirv == HIGH && ret == 0 ){ter++;sal = 0;
    buzzer_on();
    // led_on();
    Halt();
    delay(1000);
    ret = 1;
    buzzer_off();
    // led_off();
    if( ter == 1) {Pass();}
    else if (ter == 2) {Right(); }
    else if(ter == 3) { Left();}
    else if(ter == 4){ Right();}
    else if (ter == 5) {Right(); }
    else if(ter == 6) { Pass();}
    else if(ter == 7){ Right();}
    else if (ter == 8) {Pass(); }
    else if(ter == 9) { Left();}
    else if(ter == 10){ Pass();}
    else {    Pass();    }  
    }
    Halt();
    }}
     if(ter >= 10) { las++; delay(1); }
  if(las > 1800) {
      Halt();
      led_on();
      buzzer_on();
      delay(5000);
      led_off();
      buzzer_off();
      Halt();
      esp_deep_sleep_start();
      delay(10e6);
      return;
  }
  if(ret == 1 && sal > 500) { ret = 0; }
  sal++;
    }
  // delay(1);
  }
