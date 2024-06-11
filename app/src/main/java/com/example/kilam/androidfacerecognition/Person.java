package com.example.kilam.androidfacerecognition;

import java.util.Arrays;

public class Person {
    private String name;
    private String nip;
    private float[] embedding;

    public Person(String name,String nip, float[] embedding) {
        this.name = name;
        this.nip = nip;
        this.embedding = embedding;
    }

    public String getName() {
        return name;
    }
    public String getNip() {
        return nip;
    }

    public float[] getEmbedding() {
        return embedding;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Person person = (Person) o;

        if (!name.equals(person.name)) return false;
        return Arrays.equals(embedding, person.embedding);
    }

    @Override
    public int hashCode() {
        int result = name.hashCode();
        result = 31 * result + Arrays.hashCode(embedding);
        return result;
    }
}
