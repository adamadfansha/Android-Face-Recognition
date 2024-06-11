package com.example.kilam.androidfacerecognition;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;

import java.util.ArrayList;
import java.util.Arrays;

public class PersonDAO {

    private DBHelper dbHelper;

    public PersonDAO(Context context) {
        dbHelper = new DBHelper(context);
    }

    public Person addPerson(Person person) {
        SQLiteDatabase db = dbHelper.getWritableDatabase();
        ContentValues values = new ContentValues();
        values.put(DBHelper.COLUMN_NAME, person.getName());
        values.put(DBHelper.COLUMN_NIP, person.getNip());
        values.put(DBHelper.COLUMN_EMBEDDING, Arrays.toString(person.getEmbedding()));
        db.insert(DBHelper.TABLE_NAME, null, values);
        db.close();
        return person;
    }

    public void deletePerson(String name) {
        SQLiteDatabase db = dbHelper.getWritableDatabase();
        db.delete(DBHelper.TABLE_NAME, DBHelper.COLUMN_NAME + " = ?",
                new String[]{String.valueOf(name)});
        db.close();
    }

    public ArrayList<Person> getAllPersons() {
        ArrayList<Person> persons = new ArrayList<>();
        SQLiteDatabase db = dbHelper.getReadableDatabase();
        Cursor cursor = db.query(DBHelper.TABLE_NAME, null, null, null, null, null, null);
        if (cursor.moveToFirst()) {
            do {
                String name = cursor.getString(cursor.getColumnIndex(DBHelper.COLUMN_NAME));
                String nip = cursor.getString(cursor.getColumnIndex(DBHelper.COLUMN_NIP));
                String embeddingStr = cursor.getString(cursor.getColumnIndex(DBHelper.COLUMN_EMBEDDING));
                float[] embedding = convertStringToFloatArray(embeddingStr);
                persons.add(new Person(name, nip, embedding));
            } while (cursor.moveToNext());
        }
        cursor.close();
        db.close();
        return persons;
    }

    private float[] convertStringToFloatArray(String str) {
        String[] elements = str.substring(1, str.length() - 1).split(",");
        float[] floats = new float[elements.length];
        for (int i = 0; i < elements.length; i++) {
            floats[i] = Float.parseFloat(elements[i].trim());
        }
        return floats;
    }
    public boolean isNipExists(String nip) {
        SQLiteDatabase db = dbHelper.getReadableDatabase();
        Cursor cursor = db.query(DBHelper.TABLE_NAME, new String[]{DBHelper.COLUMN_NIP},
                DBHelper.COLUMN_NIP + " = ?", new String[]{nip}, null, null, null);
        boolean exists = cursor.moveToFirst();
        cursor.close();
        db.close();
        return exists;
    }
    public boolean isNameExist(String name) {
        SQLiteDatabase db = dbHelper.getReadableDatabase();
        Cursor cursor = db.query(DBHelper.TABLE_NAME, new String[]{DBHelper.COLUMN_NAME},
                DBHelper.COLUMN_NAME + " = ?", new String[]{name}, null, null, null);
        boolean exists = cursor.moveToFirst();
        cursor.close();
        db.close();
        return exists;
    }
}

